"""
День 25. Мини-чат с RAG + источниками + памятью задачи (CLI).

    python chat.py                       # сессия «default»
    python chat.py --session mcp-plan    # отдельная сессия; история и память сохраняются в sessions/

Команды:
    /state            — память задачи целиком
    /goal <текст>     — задать или сменить цель вручную
    /history          — последние сообщения
    /trace            — как обработана последняя реплика (тип, самостоятельный вопрос, поиск)
    /new              — начать сессию заново
    /exit
"""

import argparse
import os
import sys

from dialog import ChatEngine, render_turn
from rag import BASE, LLM, load_index
from retrieval import load_config


def main(args) -> None:
    llm, cfg = LLM(args.model), load_config()
    index = load_index(args.backend, "structure")
    if "answer_min" not in cfg["thresholds"][args.backend]:
        raise SystemExit(f"Нет порога «не знаю». Сначала: python calibrate.py --backend {args.backend}")
    path = os.path.join(BASE, "sessions", f"{args.session}.json")
    chat = ChatEngine(llm, index, cfg, args.backend, session_path=path)

    print(f"Мини-чат · модель {llm.model} · база {index.meta['chunks']} чанков · сессия «{args.session}»")
    if chat.turns:
        print(f"Продолжаем сохранённую сессию: {len(chat.turns)} реплик. {chat.state.compact()}")
    print("Команды: /state, /goal <текст>, /history, /trace, /new, /exit\n")

    while True:
        try:
            msg = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg:
            continue
        if msg == "/exit":
            break
        if msg == "/state":
            print(chat.state.render() + "\n")
            continue
        if msg.startswith("/goal"):
            goal = msg[5:].strip()
            if goal:
                if chat.state.goal:
                    chat.state.goal_history.append(chat.state.goal)
                chat.state.goal = goal
                chat.save()
            print(f"Цель: {chat.state.goal or 'не задана'}\n")
            continue
        if msg == "/history":
            for m in chat.history[-10:]:
                who = "Вы" if m["role"] == "user" else "Ассистент"
                print(f"  {who}: {m['content'][:150]}")
            print()
            continue
        if msg == "/trace":
            if not chat.turns:
                print("Пока не было реплик.\n")
                continue
            t = chat.turns[-1]
            print(f"  тип: {t['type']} · самостоятельный вопрос: {t['query']}")
            if t.get("retrieval"):
                r = t["retrieval"]
                print(f"  поиск: кандидатов {r['candidates']} → после порога {r['after_filter']} → {r['final']}"
                      f"{' · фильтр по ' + r['day_filter'] if r.get('day_filter') else ''}")
            print(f"  статус: {t['status']} {('(' + t['reason'] + ')') if t['reason'] else ''} · "
                  f"изменения памяти: {t['state_changes'] or 'нет'}\n")
            continue
        if msg == "/new":
            chat.reset()
            print("Новая сессия.\n")
            continue
        print("\n" + render_turn(chat.ask(msg), chat.state) + "\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Мини-чат с RAG + памятью задачи")
    ap.add_argument("--session", default="default", help="имя сессии (файл в sessions/)")
    ap.add_argument("--model", help="модель AITunnel")
    ap.add_argument("--backend", choices=["aitunnel", "local"], default="aitunnel")
    main(ap.parse_args())
