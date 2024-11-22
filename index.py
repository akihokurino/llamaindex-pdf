import argparse
import logging
import os.path
import shutil
import sys

from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from llama_index.core import GPTVectorStoreIndex
from llama_index.core import Settings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core.callbacks import CallbackManager
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.llms.langchain import LangChainLLM
from dotenv import load_dotenv


if __name__ == "__main__":
    load_dotenv()

    argp = argparse.ArgumentParser()
    argp.add_argument("-s", "--src", type=str, required=True, help="source directory")
    argp.add_argument(
        "-d", "--dest", type=str, required=True, help="destination directory"
    )
    argp.add_argument("-f", "--force", type=bool, default=False, help="force overwrite")
    args = argp.parse_args()
    src, dest, force = args.src, args.dest, args.force

    if os.path.isdir(dest):
        if force:
            shutil.rmtree(dest)
        else:
            raise Exception(
                f"Destination directory {dest} already exists. Use --force to overwrite."
            )

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    # NodeParserの設定
    # NodeParserは、テキストをチャンクに分割してノードを作成する部分を担っている
    Settings.node_parser = SimpleNodeParser()
    # Embeddingsの設定
    # テキストを埋め込みベクトルに変換する部分を担っている
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    Settings.embed_model = LangchainEmbedding(embedding, embed_batch_size=1)
    # LLMの設定
    # テキスト応答（Completion）を得るための言語モデルの部分を担っている
    Settings.llm = LangChainLLM(ChatOpenAI())
    # PromptHelperの設定
    # PromptHelperは、トークン数制限を念頭において、テキストを分割するなどの部分を担っている
    Settings.prompt_helper = PromptHelper(
        context_window=4096,  # LLMに入力するトークンの最大サイズ
        num_output=256,  # LLMから出力するトークンの最大サイズ
        chunk_overlap_ratio=0.05,  # LLMに入力する際のチャンクのオーバーラップのレート
    )
    # コールバックの設定
    # LlamaIndexの様々な処理のstart, endでコールバックを設定することができる
    # CallbackManagerにCallbackHandlerを設定することで、
    # 各CallbackHandlerのon_event_start, on_event_endが発火する
    Settings.callback_manager = CallbackManager([])

    # SimpleDirectoryReaderを使って、PDFを読み込む
    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore(),
        vector_store=SimpleVectorStore(),
        index_store=SimpleIndexStore(),
    )
    documents = SimpleDirectoryReader(input_dir=src).load_data()
    list_index = GPTVectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

    # データを保存する
    list_index.storage_context.persist(dest)
