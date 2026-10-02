"""
День 25. Проверка мини-чата на двух длинных сценариях (12 и 13 реплик).

Для каждого хода проверяется:
  - источники выведены (у ответа — хотя бы один источник; у «не знаю» — явное
    «подходящих источников нет»);
  - среди источников есть ожидаемый файл (для вопросов по базе);
  - цель сохранилась в памяти задачи — после КАЖДОГО хода;
  - тип реплики распознан как ожидалось (kb / memory / statement / offtopic);
  - в память задачи попало то, что пользователь зафиксировал;
  - ответ на «напомни цель» содержит цель и ограничения.

Важно: в промпт уходят только последние 6 сообщений. К моменту «напомни цель»
исходного сообщения с целью в окне уже нет — помнит её только память задачи.

Запуск:  python run_scenarios.py [--model gpt-5.6-luna] [--backend aitunnel|local]
"""

import argparse
import json
import os
import sys
from datetime import datetime

from dialog import WINDOW, ChatEngine, render_turn
from memory import TaskState
from rag import BASE, LLM, load_index
from retrieval import load_config


def has_all(text: str, groups: list) -> bool:
    low = text.lower()
    return all(any(v.lower() in low for v in g) for g in groups)


def check_turn(t: dict, exp: dict, goal_kw: list, state: TaskState, goal_set: bool) -> dict:
    c = {}
    c["sources_shown"] = (len(t["sources"]) >= 1) if t["status"] == "answered" else True
    if exp.get("sources"):
        c["right_source"] = any(e in s["source"] for s in t["sources"] for e in exp["sources"])
    if exp.get("status"):
        c["status"] = t["status"] == exp["status"]
    if exp.get("type"):
        c["type"] = t["type"] == exp["type"]
    if exp.get("state_has"):
        c["state"] = has_all(state.render(), exp["state_has"])
    if exp.get("answer_has"):
        c["answer"] = has_all(t["answer"], exp["answer_has"])
    if goal_set:
        c["goal_kept"] = has_all(state.goal, goal_kw)
    return c


def run_scenario(llm, index, cfg, backend, sc: dict) -> dict:
    path = os.path.join(BASE, "sessions", f"scenario_{sc['id']}.json")
    chat = ChatEngine(llm, index, cfg, backend, session_path=path)
    chat.reset()
    print(f"\n{'=' * 76}\nСценарий «{sc['title']}» — {len(sc['turns'])} реплик\n{'=' * 76}")
    rows, goal_set = [], False
    for exp in sc["turns"]:
        t = chat.ask(exp["user"])
        goal_set = goal_set or bool(chat.state.goal)
        checks = check_turn(t, exp, sc["goal_keywords"], chat.state, goal_set)
        t["checks"] = checks
        t["history_len"] = len(chat.history)
        rows.append(t)
        bad = [k for k, v in checks.items() if not v]
        src = ", ".join(sorted({s["source"].split("/")[0] for s in t["sources"]})) or "нет"
        print(f"[{t['n']:>2}] {exp['user'][:58]:<58}")
        print(f"     {t['type']:<9} → {t['status']:<8} источники: {src[:60]}"
              f"   {'✅' if not bad else '❌ ' + ', '.join(bad)}")
    return {"id": sc["id"], "title": sc["title"], "rows": rows, "final_state": chat.state.to_dict()}


def summarize(res: dict) -> dict:
    rows = res["rows"]
    def count(key):
        vals = [r["checks"][key] for r in rows if key in r["checks"]]
        return sum(vals), len(vals)
    answered = [r for r in rows if r["status"] == "answered"]
    recall = next((r for r in rows if r["type"] == "memory"), None)
    out = {k: count(k) for k in ("sources_shown", "right_source", "goal_kept", "type", "status", "state", "answer")}
    out["answered"] = (len(answered), len(rows))
    out["kb_with_quotes"] = (sum(bool(r["quotes"]) for r in answered if r["type"] == "kb"),
                             sum(r["type"] == "kb" for r in answered))
    out["recall_history"] = recall["history_len"] - 2 if recall else None
    out["weak"] = sum(bool(r.get("weak_context")) and r["status"] == "answered" for r in rows)
    out["tokens"] = sum(r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"] for r in rows)
    out["all_ok"] = all(all(r["checks"].values()) for r in rows)
    return out


LABELS = [("sources_shown", "Источники выведены в каждом ответе"),
          ("right_source", "Нужный файл среди источников"),
          ("goal_kept", "Цель сохранена после каждого хода"),
          ("type", "Тип реплики распознан верно"),
          ("status", "Ответ / «не знаю» — как ожидалось"),
          ("state", "Зафиксированное попало в память задачи"),
          ("answer", "«Напомни цель» — цель и ограничения в ответе")]


def fmt(p) -> str:
    a, b = p
    return f"{a}/{b}" + (" ✅" if a == b else " ❌") if b else "—"


def render_report(results, sums, model, index) -> str:
    head = "| Проверка | " + " | ".join(r["title"] for r in results) + " |\n|---|" + "---|" * len(results)
    lines = [f"# Мини-чат с RAG + памятью задачи: проверка на длинных сценариях\n\n"
             f"Модель: `{model}` · эмбеддинги: `{index.meta['embedder']['label']}` · окно истории в промпте: "
             f"{WINDOW} сообщений · {datetime.now():%Y-%m-%d %H:%M}\n\n## Итог\n\n{head}"]
    for key, label in LABELS:
        lines.append(f"| {label} | " + " | ".join(fmt(s[key]) for s in sums) + " |")
    lines.append("| Ответов по базе с дословными цитатами | " + " | ".join(fmt(s["kb_with_quotes"]) for s in sums) + " |")
    lines.append("| История к моменту «напомни цель», сообщений | " +
                 " | ".join(f"{s['recall_history']} (в промпте {WINDOW})" for s in sums) + " |")
    lines.append("| Ответов со слабой релевантностью (с предупреждением) | " + " | ".join(str(s["weak"]) for s in sums) + " |")
    lines.append("| Токены (вход + выход) | " + " | ".join(str(s["tokens"]) for s in sums) + " |")
    for res in results:
        st = TaskState.from_dict(res["final_state"])
        lines.append(f"\n## Сценарий «{res['title']}»\n\n**Память задачи в конце:**\n\n```\n{st.render()}\n```\n")
        if st.goal_history or st.rejected_goal_changes:
            lines.append(f"Смен цели: {len(st.goal_history)}; попыток сменить цель, отклонённых защитой: "
                         f"{st.rejected_goal_changes}\n")
        for r in res["rows"]:
            bad = [k for k, v in r["checks"].items() if not v]
            lines.append(f"### {r['n']}. Пользователь: {r['user']}\n\n"
                         f"Тип: `{r['type']}` · самостоятельный вопрос: _{r['query']}_ · "
                         f"проверки: {'✅' if not bad else '❌ ' + ', '.join(bad)}\n\n```\n{render_turn(r)}\n```\n")
    return "\n".join(lines)


def main(args) -> int:
    with open(os.path.join(BASE, "scenarios.json"), encoding="utf-8") as f:
        scenarios = json.load(f)["scenarios"]
    llm, cfg = LLM(args.model), load_config()
    index = load_index(args.backend, "structure")
    if "answer_min" not in cfg["thresholds"][args.backend]:
        raise SystemExit(f"Нет порога «не знаю». Сначала: python calibrate.py --backend {args.backend}")
    print(f"Модель: {llm.model} · эмбеддинги: {index.meta['embedder']['label']} · окно истории: {WINDOW} сообщений")

    results = [run_scenario(llm, index, cfg, args.backend, sc) for sc in scenarios]
    sums = [summarize(r) for r in results]
    print(f"\n{'=' * 76}\nИТОГ\n{'=' * 76}")
    for key, label in LABELS:
        print(f"  {label:<48} " + "   ".join(f"{fmt(s[key]):<8}" for s in sums))
    print(f"  {'Ответов по базе с дословными цитатами':<48} " + "   ".join(f"{fmt(s['kb_with_quotes']):<8}" for s in sums))
    print(f"  {'Ответов со слабой релевантностью (предупреждение)':<48} " + "   ".join(f"{s['weak']:<8}" for s in sums))
    print(f"  {'История к «напомни цель» / в промпте':<48} " +
          "   ".join(f"{s['recall_history']} / {WINDOW}" for s in sums))

    os.makedirs(os.path.join(BASE, "output"), exist_ok=True)
    path = os.path.join(BASE, "output", f"scenarios_{args.backend}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_report(results, sums, llm.model, index))
    print(f"\nПолные диалоги с источниками и памятью задачи: {os.path.relpath(path, BASE)}")
    return 0 if all(s["all_ok"] for s in sums) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Проверка мини-чата на длинных сценариях")
    ap.add_argument("--model", help="модель AITunnel")
    ap.add_argument("--backend", choices=["aitunnel", "local"], default="aitunnel")
    sys.exit(main(ap.parse_args()))
