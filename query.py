from langchain_openai.chat_models import ChatOpenAI
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.prompts.prompts import (
    QuestionAnswerPrompt,
    RefinePrompt,
    SimpleInputPrompt,
)
from llama_index.core.prompts.default_prompts import (
    DEFAULT_SIMPLE_INPUT_TMPL,
)
from dotenv import load_dotenv

# QuestionAnswerPromptの定義（デフォルトが英語なので日本語で再定義）
MY_TEXT_QA_PROMPT_TMPL = (
    "コンテキスト情報は以下のとおりです。 \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "予備知識ではなくコンテキスト情報を考慮して、質問に答えてください。:{query_str}\n"
)
# RefinePromptの定義（デフォルトが英語なので日本語で再定義）
MY_REFINE_PROMPT_TMPL = (
    "元の質問は次のとおりです: {query_str}\n"
    "既存の回答を提供しました: {existing_answer}\n"
    "以下の追加のコンテキストを使用して、既存の回答を (必要な場合のみ) 改良する機会があります。\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "新しいコンテキストを考慮して、元の答えをより良いものに改良してください。"
    "コンテキストが役に立たない場合は、元の回答を返してください。"
)

if __name__ == "__main__":
    load_dotenv()

    storage_context = StorageContext.from_defaults(persist_dir="./index")
    list_index = load_index_from_storage(storage_context=storage_context)

    ############ QueryEngineの初期化 ############
    engine = list_index.as_query_engine(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106", streaming=False),
        node_postprocessors=[],  # Node抽出後の後処理
        optimizer=None,  # 各Nodeのテキストに適用したい後処理
        response_mode=ResponseMode.COMPACT,  # レスポンス合成のモード
        # 以下は各種Promptのテンプレート定義
        text_qa_template=QuestionAnswerPrompt(MY_TEXT_QA_PROMPT_TMPL),
        refine_template=RefinePrompt(MY_REFINE_PROMPT_TMPL),
        simple_template=SimpleInputPrompt(DEFAULT_SIMPLE_INPUT_TMPL),
        streaming=False,
    )

    print("REPL mode: Type your query below. Type 'exit' to quit.")
    while True:
        query = input("Query: ")
        if query.lower() in {"exit", "quit"}:
            print("Exiting REPL. Goodbye!")
            break

        print("Completing...")
        response = engine.query(query)
        print(f"Result: {response}")
