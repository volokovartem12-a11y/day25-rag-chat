"""
День 25. Движок мини-чата: RAG + источники + память задачи.

Каждая реплика пользователя:
  1. переписывается в САМОСТОЯТЕЛЬНЫЙ вопрос с учётом истории и памяти
     («а где там хранится?» → «где хранится долговременная память в проекте дня 11?»)
     и получает тип: kb | memory | statement | offtopic;
  2. обновляет ПАМЯТЬ ЗАДАЧИ (цель, ограничения, термины, уточнения, дни) —
     с учётом типа: цель меняется только репликой-утверждением;
  3. kb        → RAG: поиск по самостоятельному вопросу и исходной реплике,
                 ответ с дословными цитатами и проверкой (день 24);
     memory    → ответ из памяти задачи и прошлых ответов, без выдумок;
     statement → подтверждение того, что зафиксировано в памяти;
     offtopic  → «не знаю» + возврат к цели;
  4. ответ ВСЕГДА сопровождается источниками: файлы и chunk_id, «память задачи»
     с источниками прошлых ответов или явное «подходящих источников нет».

В промпт уходят только последние WINDOW сообщений: длинный диалог удерживается
за счёт памяти задачи, а не за счёт всей истории.
"""

import dataclasses
import json
import os
import re
from datetime import datetime

from grounded import GROUNDED_SYSTEM, UNVERIFIED_TEXT, parse_json, suggest_clarification, validate
from memory import TaskState, update_state
from rag import build_prompt
from retrieval import MODES, mentioned_day, retrieve

WINDOW = 6  # сообщений истории в промпте
# итог и сравнение того, что уже обсудили: поиск только по источникам этого диалога
SUMMARY = re.compile(r"\b(итог\w*|подыто\w*|резюм\w*|подведи|план\w*|сравни\w*|рассмотренн\w*|обсудили)\b",
                     re.IGNORECASE)
# отсылки к предыдущему: «а где ТАМ хранится?», «как в ЭТОМ подходе?»
ANAPHORA = re.compile(r"\b(там|тут|это[мй]?|этого|этой|эту|нём|ней|него|них|такое|такой|второ\w*|перв\w*)\b",
                      re.IGNORECASE)

CONDENSE_PROMPT = """Память задачи:
{state}

Последние сообщения диалога:
{history}

Новое сообщение пользователя: {message}

Определи тип нового сообщения:
- kb — вопрос или просьба, ответ на которую нужно искать в базе документации проектов
  (включая просьбы подвести итог, сравнить, составить план по материалам);
- memory — вопрос о самом диалоге: какая цель, что договорились, что уже выяснили;
- statement — пользователь только сообщает цель, ограничение, термин или решение, без вопроса;
- offtopic — сообщение не относится ни к цели диалога, ни к базе.
Для kb перепиши сообщение в САМОСТОЯТЕЛЬНЫЙ поисковый вопрос: подставь из истории и памяти
всё, на что ссылаются «там», «это», «а во втором» (номер дня, название проекта, термин);
если вопрос задан в рамках цели диалога, добавь ключевые термины цели (например, «MCP», «контекст»),
чтобы поиск понимал тему; если просят итог или план — сформулируй вопрос по цели диалога.
Верни ТОЛЬКО JSON: {{"type": "kb|memory|statement|offtopic", "query": "самостоятельный вопрос"}}"""

CHAT_RULES = (
    "\n\nТы ведёшь диалог. Учитывай память задачи: держи в фокусе цель диалога, соблюдай "
    "ограничения и используй термины так, как договорились с пользователем. Если пользователь "
    "просит итог или план — опирайся на фрагменты контекста и всё равно приводи цитаты."
)


def _add(total: dict, usage: dict) -> None:
    for k in total:
        total[k] += usage.get(k, 0)


class ChatEngine:
    def __init__(self, llm, index, cfg: dict, backend: str, session_path: str = None, window: int = WINDOW):
        self.llm, self.index, self.cfg, self.backend, self.window = llm, index, cfg, backend, window
        self.session_path = session_path
        self.history, self.turns, self.state = [], [], TaskState()
        self.known_days = sorted({int(m.group(1)) for c in index.chunks
                                  for m in [re.match(r"day(\d+)-", c["source"])] if m})
        if session_path and os.path.exists(session_path):
            self.load()

    # ---------- сессия ----------

    def save(self) -> None:
        if not self.session_path:
            return
        os.makedirs(os.path.dirname(self.session_path) or ".", exist_ok=True)
        with open(self.session_path, "w", encoding="utf-8") as f:
            json.dump({"saved": datetime.now().isoformat(timespec="seconds"), "state": self.state.to_dict(),
                       "history": self.history, "turns": self.turns}, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        with open(self.session_path, encoding="utf-8") as f:
            data = json.load(f)
        self.state = TaskState.from_dict(data.get("state", {}))
        self.history, self.turns = data.get("history", []), data.get("turns", [])

    def reset(self) -> None:
        self.history, self.turns, self.state = [], [], TaskState()
        self.save()

    # ---------- шаги обработки ----------

    def _recent(self) -> str:
        msgs = self.history[-self.window:]
        return "\n".join(f"{'Пользователь' if m['role'] == 'user' else 'Ассистент'}: {m['content'][:400]}"
                         for m in msgs) or "(диалог только начался)"

    def _condense(self, message: str, usage: dict) -> tuple:
        raw, u = self.llm.chat("Ты разбираешь реплики диалога.", CONDENSE_PROMPT.format(
            state=self.state.render(), history=self._recent(), message=message))
        _add(usage, u)
        obj = parse_json(raw) or {}
        kind = obj.get("type") if obj.get("type") in ("kb", "memory", "statement", "offtopic") else "kb"
        query = str(obj.get("query") or message).strip()
        # страховка: реплика ссылается на предыдущее («там», «это»), а день в вопросе потерялся — подставляем.
        # Без отсылки не подставляем: пользователь мог сменить тему.
        if (kind == "kb" and ANAPHORA.search(message) and mentioned_day(query) is None
                and mentioned_day(message) is None and self.state.focus_days):
            query = f"{query} (проект дня {self.state.focus_days[-1]})"
        return kind, query

    def _kb_answer(self, message: str, query: str, turn: dict) -> None:
        usage = turn["usage"]
        day = mentioned_day(query)
        if day is not None and day not in self.known_days:
            turn.update(status="unknown", reason="no_day", answer="Не знаю: такого проекта в базе нет.",
                        clarification=f"В базе нет проекта дня {day}. Есть дни: "
                                      f"{', '.join(map(str, self.known_days))}. Какой из них вас интересует?")
            return
        gate = self.cfg["thresholds"][self.backend].get("answer_min", 0.0)
        turn["top_score"] = round(max(self.index.search(q, 1)[0]["score"] for q in (query, message)), 4)
        turn["gate"] = gate
        if day is None and turn["top_score"] < gate:
            if not self.state.goal:
                # холодный старт: контекста диалога нет — слабая близость означает «не знаю»
                turn.update(status="unknown", reason="gate",
                            answer="Не знаю: в базе не нашлось достаточно релевантных фрагментов.",
                            clarification=suggest_clarification(self.index, query))
                return
            # цель диалога известна и реплика — вопрос по базе: ищем, но предупреждаем;
            # ответ всё равно обязан опираться на дословные цитаты, иначе «не знаю»
            turn["weak_context"] = True

        mode, scope = MODES["filter"], None
        if SUMMARY.search(message):
            scope = {s["source"] for t in self.turns if t["type"] == "kb" and t["status"] == "answered"
                     for s in t["sources"]}
            if scope:  # итог по тому, что уже обсудили: больше фрагментов, только источники диалога
                mode = dataclasses.replace(MODES["filter"], k_after=6)
                turn["weak_context"] = False  # релевантность задана самим диалогом
        r = retrieve(self.index, query, mode, self.backend, self.cfg, extra_queries=[message],
                     allowed_sources=scope or None)
        turn["retrieval"] = {"candidates": r.trace["candidates"], "after_filter": r.trace["after_filter"],
                             "final": r.trace["final"], "day_filter": r.trace.get("day_filter"),
                             "dialog_scope": len(scope) if scope else 0}
        if not r.chunks:
            turn.update(status="unknown", reason="empty", answer="Не знаю: подходящих фрагментов не найдено.",
                        clarification=suggest_clarification(self.index, query))
            return

        prompt = (f"Память задачи:\n{self.state.render()}\n\nПоследние сообщения диалога:\n{self._recent()}\n\n"
                  + build_prompt(f"{message}\n(самостоятельная формулировка: {query})", r.chunks))
        system = GROUNDED_SYSTEM + CHAT_RULES
        raw, u = self.llm.chat(system, prompt)
        _add(usage, u)
        parsed, problems = validate(parse_json(raw), r.chunks)
        turn["attempts"] = 1
        if problems:
            fb = "Ответ не прошёл проверку:\n- " + "\n- ".join(problems) + "\nИсправь и верни только JSON."
            raw2, u2 = self.llm.chat(system, prompt, history=[{"role": "assistant", "content": raw},
                                                              {"role": "user", "content": fb}])
            _add(usage, u2)
            turn["attempts"] = 2
            parsed2, problems2 = validate(parse_json(raw2), r.chunks)
            if parsed2 is not None and (parsed is None or len(problems2) <= len(problems)):
                parsed, problems = parsed2, problems2

        if parsed is None or (parsed["status"] == "answered" and (not parsed["quotes"] or not parsed["sources"])):
            turn.update(status="unknown", reason="unverified", answer=UNVERIFIED_TEXT,
                        clarification=suggest_clarification(self.index, query))
            return
        turn["rejected_quotes"] = len(parsed["rejected"])
        if parsed["status"] == "unknown":
            turn.update(status="unknown", reason="model", answer=parsed["answer"] or "Не знаю.",
                        clarification=parsed["clarification"] or suggest_clarification(self.index, query))
            return
        turn.update(status="answered", answer=parsed["answer"], quotes=parsed["quotes"],
                    sources=[{"type": "kb", **s} for s in parsed["sources"]])

    def _memory_answer(self, turn: dict) -> None:
        """Ответ о самом диалоге — только из памяти задачи и прошлых ответов, без модели."""
        s = self.state
        lines = [f"Цель: {s.goal or 'пока не задана'}"]
        if s.constraints:
            lines.append("Ограничения: " + "; ".join(s.constraints))
        if s.terms:
            lines.append("Термины: " + "; ".join(f"«{k}» = {v}" for k, v in s.terms.items()))
        if s.clarified:
            lines.append("Уточнено: " + "; ".join(s.clarified[-5:]))
        done = [t for t in self.turns if t["status"] == "answered" and t["type"] == "kb"]
        if done:
            lines.append("Уже выяснили:")
            lines += [f"  — {t['query'][:90]}" for t in done[-6:]]
        past = []
        for t in done[-6:]:
            for src in t["sources"]:
                if src["source"] not in [p["source"] for p in past]:
                    past.append({"type": "history", "source": src["source"], "section": src["section"],
                                 "chunk_id": src["chunk_id"], "turn": t["n"]})
        turn.update(status="answered", answer="\n".join(lines),
                    sources=[{"type": "memory", "source": "память задачи", "section": "цель, ограничения, термины",
                              "chunk_id": "task_state"}] + past)

    def _statement_answer(self, changed: dict, turn: dict) -> None:
        parts = []
        if "goal" in changed:
            parts.append(f"цель: «{changed['goal']}»")
        for c in changed.get("constraints_add", []):
            parts.append(f"ограничение: «{c}»")
        for k, v in changed.get("terms_add", {}).items():
            parts.append(f"термин «{k}» = {v}")
        for c in changed.get("clarified_add", []):
            parts.append(f"уточнение: «{c}»")
        text = ("Зафиксировал в памяти задачи — " + "; ".join(parts) + "." if parts
                else "Принял. Новых целей, ограничений или терминов в сообщении не нашёл.")
        turn.update(status="answered", answer=text + " Задавайте вопрос — отвечу по базе с источниками.",
                    sources=[{"type": "memory", "source": "память задачи", "section": "обновлена этой репликой",
                              "chunk_id": "task_state"}])

    # ---------- один ход ----------

    def ask(self, message: str) -> dict:
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        last_answer = next((m["content"] for m in reversed(self.history) if m["role"] == "assistant"), "")
        # сначала тип реплики, потом память: вопрос «напомни цель» не должен менять цель
        kind, query = self._condense(message, usage)
        changed, u = update_state(self.llm, self.state, message, last_answer, kind)
        _add(usage, u)

        turn = {"n": len(self.turns) + 1, "user": message, "type": kind, "query": query, "status": "",
                "reason": "", "answer": "", "sources": [], "quotes": [], "clarification": "",
                "state_changes": changed, "usage": usage, "attempts": 0}
        if kind == "memory":
            self._memory_answer(turn)
        elif kind == "statement":
            self._statement_answer(changed, turn)
        elif kind == "offtopic":
            turn.update(status="unknown", reason="offtopic", answer="Не знаю: вопрос не относится к базе и к цели диалога.",
                        clarification=f"Вернёмся к цели: «{self.state.goal}»? Спросите, что из неё разобрать дальше."
                        if self.state.goal else suggest_clarification(self.index, message))
        else:
            self._kb_answer(message, query, turn)

        turn["state"] = self.state.to_dict()
        self.turns.append(turn)
        self.history += [{"role": "user", "content": message},
                         {"role": "assistant", "content": turn["answer"]}]
        self.save()
        return turn


def render_turn(t: dict, state: TaskState = None) -> str:
    """Как ответ видит пользователь. Строка «Источники» есть всегда."""
    lines = [f"Ассистент: {t['answer']}"]
    if t.get("weak_context") and t["status"] == "answered":
        lines.append(f"⚠ Контекст найден со слабой релевантностью (близость {t['top_score']} < порога "
                     f"{t['gate']}) — ответ опирается только на дословные цитаты ниже, проверьте их.")
    if t.get("clarification"):
        lines.append(t["clarification"])
    lines.append("\nИсточники:")
    if t["sources"]:
        for i, s in enumerate(t["sources"], 1):
            if s["type"] == "kb":
                lines.append(f"  [{s['n']}] {s['source']} — {s['section']}  ({s['chunk_id']})")
            elif s["type"] == "memory":
                lines.append(f"  • {s['source']} — {s['section']}")
            else:
                lines.append(f"  • {s['source']} — {s['section']}  (из ответа №{s['turn']})")
    else:
        lines.append("  — подходящих источников в базе нет")
    if t["quotes"]:
        lines.append("Цитаты:")
        lines += [f"  [{q['n']}] «{q['text']}»" for q in t["quotes"]]
    if state is not None:
        lines.append(f"\n{state.compact()}")
    return "\n".join(lines)
