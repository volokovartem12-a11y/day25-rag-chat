"""
День 25. Память задачи (task state).

Хранится отдельно от истории сообщений и обновляется после КАЖДОЙ реплики
пользователя:
  goal         — цель диалога;
  constraints  — ограничения («бюджет на токены жёсткий», «без SDK»);
  terms        — зафиксированные термины («факты» = Sticky Facts из дня 10);
  clarified    — что пользователь уже уточнил или решил;
  focus_days   — о каких проектах (днях челленджа) сейчас идёт речь.

Защита цели: модель-экстрактор может предложить новую цель, но код примет её
только из реплики-утверждения («Цель: …», «хочу …») или если цели ещё нет.
Вопросы — в том числе вопросы О цели («напомни, какая цель?») — и посторонние
реплики цель не меняют никогда. Именно это не даёт ассистенту «потерять нить».
"""

import json
import re
from dataclasses import asdict, dataclass, field

from retrieval import mentioned_day

GOAL_CUES = re.compile(r"\bцел[ьиюея]|\bзадач|\bхочу\b|\bнужно\b|\bвыбираю\b|\bразобраться\b|\bпереключ|\bтеперь будем\b",
                       re.IGNORECASE)

EXTRACT_PROMPT = """Ты ведёшь «память задачи» диалога с ассистентом по базе документации проектов.

Текущая память (JSON):
{state}

Последний ответ ассистента:
{last_answer}

Новое сообщение пользователя:
{message}

Определи, что НОВОГО сообщил пользователь, и верни ТОЛЬКО JSON:
{{"goal": null или "цель диалога — ТОЛЬКО если пользователь явно ставит или меняет цель",
 "constraints_add": ["новые ограничения и требования пользователя"],
 "terms_add": {{"термин": "как пользователь договорился его понимать"}},
 "clarified_add": ["что пользователь уточнил или решил (не его вопросы!)"],
 "focus_days": [номера дней челленджа, о которых сейчас речь, или пусто]}}
Не повторяй то, что уже есть в памяти. Вопрос пользователя — не уточнение и не ограничение."""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower().rstrip(".!"))


@dataclass
class TaskState:
    goal: str = ""
    constraints: list = field(default_factory=list)
    terms: dict = field(default_factory=dict)
    clarified: list = field(default_factory=list)
    focus_days: list = field(default_factory=list)
    goal_history: list = field(default_factory=list)   # как менялась цель (для отчёта)
    rejected_goal_changes: int = 0                     # сколько раз защита не дала сменить цель

    def render(self) -> str:
        """Блок для промпта модели."""
        lines = [f"Цель диалога: {self.goal or 'не задана'}"]
        if self.constraints:
            lines.append("Ограничения: " + "; ".join(self.constraints))
        if self.terms:
            lines.append("Термины: " + "; ".join(f"«{k}» = {v}" for k, v in self.terms.items()))
        if self.clarified:
            lines.append("Уже уточнено: " + "; ".join(self.clarified[-6:]))
        if self.focus_days:
            lines.append("Сейчас речь о днях: " + ", ".join(map(str, self.focus_days)))
        return "\n".join(lines)

    def compact(self) -> str:
        """Одна строка для чата."""
        parts = [f"🎯 {self.goal[:80] or 'цель не задана'}"]
        if self.constraints:
            parts.append(f"ограничений {len(self.constraints)}")
        if self.terms:
            parts.append(f"терминов {len(self.terms)}")
        if self.focus_days:
            parts.append("дни " + ",".join(map(str, self.focus_days)))
        return " · ".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TaskState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def apply_update(state: TaskState, upd: dict, message: str, kind: str = "statement") -> dict:
    """Применяет изменения экстрактора с защитой цели. kind — тип реплики (kb/memory/statement/offtopic).
    Возвращает, что реально изменилось."""
    changed = {}
    if kind == "offtopic":
        upd = {"goal": upd.get("goal")}  # из посторонней реплики в память ничего не берём
    goal = upd.get("goal")
    if isinstance(goal, str) and goal.strip() and _norm(goal) != _norm(state.goal):
        allowed = (not state.goal and kind != "offtopic") or (kind == "statement" and GOAL_CUES.search(message))
        if allowed:
            if state.goal:
                state.goal_history.append(state.goal)
            state.goal = goal.strip()
            changed["goal"] = state.goal
        else:
            state.rejected_goal_changes += 1
            changed["goal_change_rejected"] = goal.strip()

    for key, target in (("constraints_add", state.constraints), ("clarified_add", state.clarified)):
        known = {_norm(x) for x in target}
        for item in upd.get(key) or []:
            if isinstance(item, str) and item.strip() and _norm(item) not in known:
                target.append(item.strip())
                known.add(_norm(item))
                changed.setdefault(key, []).append(item.strip())

    for k, v in (upd.get("terms_add") or {}).items():
        if isinstance(k, str) and isinstance(v, str) and k.strip() and state.terms.get(k.strip()) != v.strip():
            state.terms[k.strip()] = v.strip()
            changed.setdefault("terms_add", {})[k.strip()] = v.strip()

    days = [d for d in (upd.get("focus_days") or []) if isinstance(d, int)]
    explicit = mentioned_day(message)
    if explicit is not None and explicit not in days:
        days.append(explicit)  # день, названный в реплике, учитываем даже если экстрактор его пропустил
    if days and days != state.focus_days:
        state.focus_days = days
        changed["focus_days"] = days
    return changed


def update_state(llm, state: TaskState, message: str, last_answer: str, kind: str = "statement") -> tuple:
    """LLM извлекает новое из реплики; код применяет с защитой цели."""
    raw, usage = llm.chat("Ты аккуратно ведёшь память задачи диалога.", EXTRACT_PROMPT.format(
        state=json.dumps({k: v for k, v in state.to_dict().items()
                          if k not in ("goal_history", "rejected_goal_changes")}, ensure_ascii=False),
        last_answer=(last_answer or "—")[:600], message=message))
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        upd = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        upd = {}
    return apply_update(state, upd if isinstance(upd, dict) else {}, message, kind), usage
