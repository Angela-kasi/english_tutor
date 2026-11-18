下面是整理后的版本。教学提示词改成了简短英文，中文注释都以 `#` 开头，标点使用英文半角。每个文件单独保存；`config.py` 和 `app.py` 放在 `english_tutor` 目录，其余模块放在 `english_tutor/modules` 目录。

## 1. `config.py`

```python
import os

# App settings
APP_TITLE = "高三英语智能助教"
APP_SUBTITLE = "作文批改 | 长难句解析 | 单词小测 | 学习诊断 | 本地知识库"

MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.getenv(
    "DEEPSEEK_BASE_URL",
    "https://api.deepseek.com",
)

# Project paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
CORPUS_DIR = os.path.join(DATA_DIR, "corpus")
LOG_PATH = os.path.join(DATA_DIR, "learner_log.json")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")

QUIZ_COUNT = 5
MAX_LOG_ENTRIES = 5000
MAX_DIAGNOSIS_RECORDS = 100

# Teaching prompts
ESSAY_PROMPT = """
You are a careful English teacher for Chinese high school students.
Give constructive feedback in Chinese. Include:
1. Overall comments.
2. An estimated score if a rubric is provided.
3. Content, language, and organization feedback.
4. Three priority corrections with reasons.
5. A revised version that preserves the student's meaning.
Do not invent requirements. State uncertainty clearly.
"""

SENTENCE_PROMPT = """
You are an English grammar teacher for Chinese high school students.
Explain the sentence in Chinese. Include:
1. Main clause.
2. Meaning groups separated with slashes.
3. Clauses, non-finite verbs, and special structures.
4. A natural Chinese translation.
5. One practice question.
Mention ambiguity when needed.
"""

QUIZ_PROMPT = """
Create multiple-choice vocabulary questions for Chinese high school students.
Return only a JSON array. Each item must contain:
word, question, options, answer, explanation.
Options must have keys A, B, C, and D. There must be one correct answer.
Write the explanation in Chinese. Do not use Markdown fences.
"""

DIAGNOSIS_PROMPT = """
You are a careful high school English teacher.
Review the learning records and respond in Chinese.
Summarize observed evidence, possible strengths, areas to improve,
and three practical suggestions for next week.
Use only the evidence provided. Do not label the student.
"""

RAG_ANSWER_PROMPT = """
Answer the student's question in Chinese.
Use the provided local knowledge excerpts first.
If the excerpts do not support an answer, say so clearly.
Do not invent sources or quotations.
"""
```

## 2. `api_utils.py`

```python
import json
import os
from datetime import datetime
from typing import Any

import streamlit as st
from openai import OpenAI

from config import (
    DEEPSEEK_BASE_URL,
    LOG_PATH,
    MAX_LOG_ENTRIES,
    MODEL_NAME,
)


def get_client() -> OpenAI:
    # Read the API key from an environment variable or Streamlit secrets.
    api_key = os.getenv("DEEPSEEK_API_KEY", "")

    if not api_key:
        try:
            api_key = st.secrets.get("DEEPSEEK_API_KEY", "")
        except Exception:
            api_key = ""

    if not api_key:
        raise RuntimeError(
            "DeepSeek API key is missing. "
            "Add it to .streamlit/secrets.toml."
        )

    return OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
    )


def ask_model(
    system_prompt: str,
    user_content: str,
    temperature: float = 0.3,
) -> str:
    # Send a request to DeepSeek and return the response text.
    client = get_client()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
    )

    return response.choices[0].message.content or "No response was returned."


def log_learning_event(
    feature: str,
    user_input: str,
    assistant_output: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    # Save each interaction to the local JSON log.
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    try:
        with open(LOG_PATH, "r", encoding="utf-8") as file:
            records = json.load(file)
        if not isinstance(records, list):
            records = []
    except (FileNotFoundError, json.JSONDecodeError):
        records = []

    records.append(
        {
            "timestamp": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "feature": feature,
            "user_input": user_input,
            "assistant_output": assistant_output,
            "metadata": metadata or {},
        }
    )

    records = records[-MAX_LOG_ENTRIES:]

    with open(LOG_PATH, "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)


def read_learning_log() -> list[dict[str, Any]]:
    # Return an empty list when the log does not exist.
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as file:
            records = json.load(file)
        return records if isinstance(records, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []
```

## 3. `essay_scorer.py`

```python
from config import ESSAY_PROMPT
from modules.api_utils import ask_model, log_learning_event


def score_essay(
    topic: str,
    essay: str,
    score_total: int = 25,
    rubric: str = "",
) -> str:
    # Prepare the essay information for the model.
    request = (
        f"Topic: {topic or 'Not provided'}\n"
        f"Maximum score: {score_total}\n"
        f"Rubric: {rubric or 'Not provided'}\n"
        f"Essay:\n{essay}"
    )

    result = ask_model(ESSAY_PROMPT, request)

    # Save the essay and feedback as learning evidence.
    log_learning_event(
        feature="essay_feedback",
        user_input=request,
        assistant_output=result,
        metadata={"score_total": score_total},
    )

    return result
```

## 4. `sentence_parser.py`

```python
from config import SENTENCE_PROMPT
from modules.api_utils import ask_model, log_learning_event


def parse_sentence(sentence: str) -> str:
    # Ask the model to analyze the sentence.
    result = ask_model(SENTENCE_PROMPT, sentence)

    # Save the sentence and explanation.
    log_learning_event(
        feature="sentence_analysis",
        user_input=sentence,
        assistant_output=result,
    )

    return result
```

## 5. `quiz_builder.py`

```python
import json
import re
from typing import Any

from config import QUIZ_COUNT, QUIZ_PROMPT
from modules.api_utils import ask_model, log_learning_event


def build_quiz(
    words: str,
    count: int = QUIZ_COUNT,
) -> list[dict[str, Any]]:
    # Ask the model to generate a JSON quiz.
    raw_text = ask_model(
        system_prompt=QUIZ_PROMPT,
        user_content=f"Words: {words}\nQuestion count: {count}",
        temperature=0.2,
    )

    # Remove Markdown fences if the model adds them.
    cleaned_text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        raw_text.strip(),
        flags=re.IGNORECASE,
    )

    try:
        questions = json.loads(cleaned_text)
    except json.JSONDecodeError as error:
        raise ValueError("The generated quiz is not valid JSON.") from error

    if not isinstance(questions, list):
        raise ValueError("The generated quiz must be a JSON list.")

    return questions


def save_quiz_result(
    words: str,
    answers: dict[int, str],
    questions: list[dict[str, Any]],
) -> tuple[int, int]:
    # Compare student answers with the answer key.
    correct_count = 0
    answer_details = []

    for index, question in enumerate(questions):
        chosen = answers.get(index, "Unanswered")
        correct_answer = str(question.get("answer", "")).strip().upper()
        is_correct = chosen == correct_answer

        if is_correct:
            correct_count += 1

        answer_details.append(
            {
                "word": question.get("word", ""),
                "chosen": chosen,
                "correct_answer": correct_answer,
                "is_correct": is_correct,
            }
        )

    # Save the result and answer details.
    log_learning_event(
        feature="vocabulary_quiz",
        user_input=words,
        assistant_output=f"{correct_count}/{len(questions)} correct",
        metadata={"answers": answer_details},
    )

    return correct_count, len(questions)
```

## 6. `learner_diag.py`

```python
import json

from config import DIAGNOSIS_PROMPT, MAX_DIAGNOSIS_RECORDS
from modules.api_utils import (
    ask_model,
    log_learning_event,
    read_learning_log,
)


def diagnose_learner() -> str:
    # Load recent learning records.
    records = read_learning_log()

    if not records:
        return "No learning records found. Complete a learning activity first."

    recent_records = records[-MAX_DIAGNOSIS_RECORDS:]
    evidence = json.dumps(recent_records, ensure_ascii=False, indent=2)

    # Ask the model to summarize the evidence.
    result = ask_model(
        system_prompt=DIAGNOSIS_PROMPT,
        user_content=f"Recent learning records:\n{evidence}",
    )

    # Save the diagnosis as another learning event.
    log_learning_event(
        feature="learning_diagnosis",
        user_input=f"Based on {len(recent_records)} recent records",
        assistant_output=result,
        metadata={"records_considered": len(recent_records)},
    )

    return result
```

## 7. `rag_retrieval.py`

```python
import os
import re

from config import CORPUS_DIR, RAG_ANSWER_PROMPT
from modules.api_utils import ask_model, log_learning_event


def load_documents() -> list[dict[str, str]]:
    # Read UTF-8 text and Markdown files from the corpus folder.
    os.makedirs(CORPUS_DIR, exist_ok=True)
    documents = []

    for filename in os.listdir(CORPUS_DIR):
        if not filename.lower().endswith((".txt", ".md")):
            continue

        file_path = os.path.join(CORPUS_DIR, filename)

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
        except (OSError, UnicodeDecodeError):
            continue

        paragraphs = re.split(r"\n\s*\n", content)

        for paragraph in paragraphs:
            paragraph = paragraph.strip()

            for start in range(0, len(paragraph), 900):
                chunk = paragraph[start : start + 900].strip()
                if chunk:
                    documents.append(
                        {
                            "source": filename,
                            "text": chunk,
                        }
                    )

    return documents


def extract_terms(text: str) -> set[str]:
    # Extract basic English words and Chinese text groups.
    return set(
        re.findall(
            r"[a-zA-Z]{2,}|[\u4e00-\u9fff]{2,}",
            text.lower(),
        )
    )


def retrieve(
    question: str,
    limit: int = 4,
) -> list[dict[str, str]]:
    # Rank document chunks by the number of matching terms.
    question_terms = extract_terms(question)
    scored_documents = []

    for document in load_documents():
        document_terms = extract_terms(document["text"])
        score = len(question_terms & document_terms)

        if score > 0:
            scored_documents.append((score, document))

    scored_documents.sort(key=lambda item: item[0], reverse=True)

    return [
        document
        for score, document in scored_documents[:limit]
    ]


def answer_from_corpus(
    question: str,
) -> tuple[str, list[dict[str, str]]]:
    # Find relevant local text before asking the model.
    matches = retrieve(question)

    if matches:
        context = "\n\n".join(
            f"Source: {item['source']}\nText: {item['text']}"
            for item in matches
        )
    else:
        context = "No matching local materials were found."

    result = ask_model(
        system_prompt=RAG_ANSWER_PROMPT,
        user_content=f"Local materials:\n{context}\n\nQuestion:\n{question}",
    )

    sources = sorted({item["source"] for item in matches})

    # Save the question, response, and matched sources.
    log_learning_event(
        feature="local_knowledge_search",
        user_input=question,
        assistant_output=result,
        metadata={"sources": sources},
    )

    return result, matches
```

## 8. `app.py`

```python
import os

import streamlit as st

from config import APP_SUBTITLE, APP_TITLE, OUTPUT_DIR, QUIZ_COUNT
from modules.essay_scorer import score_essay
from modules.learner_diag import diagnose_learner
from modules.quiz_builder import build_quiz, save_quiz_result
from modules.rag_retrieval import answer_from_corpus
from modules.sentence_parser import parse_sentence


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📘",
    layout="wide",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

st.title(f"📘 {APP_TITLE}")
st.caption(APP_SUBTITLE)
st.info(
    "Use anonymous student work when possible. "
    "Review model feedback before using it for grading."
)

essay_tab, sentence_tab, quiz_tab, diagnosis_tab, rag_tab = st.tabs(
    [
        "作文批改",
        "长难句解析",
        "单词小测",
        "学习诊断",
        "本地知识库",
    ]
)


with essay_tab:
    st.subheader("作文批改")

    with st.form("essay_form"):
        topic = st.text_input("作文题目")
        score_total = st.number_input(
            "作文满分",
            min_value=1,
            max_value=100,
            value=25,
        )
        rubric = st.text_area("评分标准")
        essay = st.text_area("学生作文", height=220)
        essay_submitted = st.form_submit_button(
            "生成批改建议",
            type="primary",
        )

    if essay_submitted:
        if not essay.strip():
            st.warning("请先输入作文内容。")
        else:
            with st.spinner("正在生成反馈..."):
                try:
                    result = score_essay(
                        topic=topic,
                        essay=essay,
                        score_total=int(score_total),
                        rubric=rubric,
                    )
                    st.markdown(result)
                except Exception as error:
                    st.error(f"批改失败: {error}")


with sentence_tab:
    st.subheader("长难句解析")

    with st.form("sentence_form"):
        sentence = st.text_area("英文句子", height=130)
        sentence_submitted = st.form_submit_button(
            "拆解句子",
            type="primary",
        )

    if sentence_submitted:
        if not sentence.strip():
            st.warning("请先输入英文句子。")
        else:
            with st.spinner("正在分析句子..."):
                try:
                    st.markdown(parse_sentence(sentence))
                except Exception as error:
                    st.error(f"解析失败: {error}")


with quiz_tab:
    st.subheader("单词小测")
    st.caption(f"默认生成 {QUIZ_COUNT} 道四选一题。")

    words = st.text_area(
        "本次复习词汇",
        placeholder="例如: adapt, approach, benefit",
    )

    if st.button("生成小测", type="primary", key="make_quiz"):
        if not words.strip():
            st.warning("请先输入词汇。")
        else:
            with st.spinner("正在生成题目..."):
                try:
                    st.session_state["quiz_questions"] = build_quiz(words)
                    st.session_state["quiz_words"] = words
                except Exception as error:
                    st.error(f"生成失败: {error}")

    questions = st.session_state.get("quiz_questions", [])

    if questions:
        with st.form("quiz_answer_form"):
            selected_answers = {}

            for index, question in enumerate(questions):
                st.markdown(
                    f"**第 {index + 1} 题: "
                    f"{question.get('word', '词汇')}**"
                )
                st.write(question.get("question", ""))

                options = question.get("options", {})
                choices = [
                    f"{letter}. {options.get(letter, '')}"
                    for letter in ("A", "B", "C", "D")
                ]

                selected = st.radio(
                    "选择答案",
                    choices,
                    index=None,
                    key=f"quiz_{index}",
                    label_visibility="collapsed",
                )

                selected_answers[index] = (
                    selected.split(".", 1)[0]
                    if selected
                    else "Unanswered"
                )

            quiz_submitted = st.form_submit_button("提交答案")

        if quiz_submitted:
            correct, total = save_quiz_result(
                words=st.session_state.get("quiz_words", ""),
                answers=selected_answers,
                questions=questions,
            )

            st.success(f"答对 {correct}/{total} 题。")

            for index, question in enumerate(questions):
                with st.expander(f"第 {index + 1} 题解析"):
                    st.write(f"正确答案: {question.get('answer', '')}")
                    st.write(question.get("explanation", ""))


with diagnosis_tab:
    st.subheader("学习诊断")
    st.write("根据本机保存的学习记录生成总结和建议。")

    if st.button("生成学习诊断", type="primary"):
        with st.spinner("正在整理学习记录..."):
            try:
                st.markdown(diagnose_learner())
            except Exception as error:
                st.error(f"诊断失败: {error}")


with rag_tab:
    st.subheader("本地知识库检索")
    st.write("将 UTF-8 编码的 .txt 或 .md 资料放入 data/corpus。")

    question = st.text_area("学生的问题", height=100)

    if st.button("检索并回答", type="primary", key="rag_query"):
        if not question.strip():
            st.warning("请先输入问题。")
        else:
            with st.spinner("正在搜索资料..."):
                try:
                    answer, sources = answer_from_corpus(question)
                    st.markdown(answer)

                    if sources:
                        with st.expander("查看命中的资料"):
                            for source in sources:
                                st.caption(f"来源: {source['source']}")
                                st.write(source["text"])
                    else:
                        st.caption("没有找到关键词匹配的本地资料。")
                except Exception as error:
                    st.error(f"检索失败: {error}")


st.divider()
st.caption(
    "学习记录保存在 data/learner_log.json。"
    "调用 DeepSeek 需要有效的 API 密钥和网络连接。"
)
```

## 9. `requirements.txt`

```python
streamlit>=1.35.0
openai>=1.30.0
```

保存后，在项目目录下创建 `data/corpus`、`output` 和 `.streamlit` 文件夹。然后在 `.streamlit` 中新建 `secrets.toml`，写入：

```python
DEEPSEEK_API_KEY = "替换成你的 DeepSeek API 密钥"
```

进入 `english_tutor` 项目目录后，运行：

```python
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
