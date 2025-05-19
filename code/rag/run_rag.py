from uuid import uuid4

import gradio as gr
from src.rag_pipeline import RagPipeline

rag_pipeline = RagPipeline(chunk_size=500, chunk_overlap=50, use_tools=False)


def clear_history():
    new_id = uuid4()
    rag_pipeline.clear_vector_store()
    print(f"New thread_id: {new_id}")
    return str(new_id), None


def clear_document_list():
    rag_pipeline.clear_vector_store()
    return [], gr.Dataset(sample_labels=[], type="values")


def add_message(history, message, thread_id):
    if message:
        history.append({"role": "user", "content": message})
        response = rag_pipeline.query(message, thread_id=thread_id)
        bot_message = response["messages"][-1].content
        history.append({"role": "assistant", "content": bot_message})

    return history, gr.Textbox(value=None, interactive=False)


def add_document(new_docs, doc_list):
    rag_pipeline.add_documents(new_docs)
    doc_list.extend([doc.split("/")[-1] for doc in new_docs])
    return doc_list, gr.Dataset(sample_labels=doc_list, type="values")


with gr.Blocks(title="🤖 FlexBot: Ask me anything! 🤖") as demo:
    uuid_state = gr.State(value=str(uuid4()))
    doc_list = gr.State(value=[])

    title = """
    <center>
    <h1>🤖 FlexBot: Ask me anything! 🤖</h1>
    </center>
    <h2>FlexBot can answer questions based on the provided documents.</h2>
    """
    with gr.Row():
        gr.HTML(title)
    chatbot = gr.Chatbot(elem_id="chatbot", type="messages")
    chat_input = gr.Textbox(
        interactive=True,
        placeholder="Ask me anything!",
        show_label=False,
        submit_btn=True,
    )

    gr.Markdown("")
    gr.Markdown("")
    gr.Markdown("")
    gr.Markdown("## RAG documents")
    with gr.Row():
        file_table = gr.Examples(
            [],
            doc_list,
            label="Knowledge base content",
        )
    with gr.Row():
        with gr.Column(scale=2):
            doc_input = gr.UploadButton(
                label="Upload file documents",
                file_types=[
                    ".txt",
                    ".pdf",
                    ".csv",
                    ".doc",
                    ".docx",
                    ".html",
                    ".htm",
                ],
                file_count="multiple",
            )
        with gr.Column(scale=1):
            clear_doc_button = gr.ClearButton(value="Clear all documents")

    chatbot.clear(clear_history, outputs=[uuid_state, chatbot])

    chat_msg = chat_input.submit(
        add_message, [chatbot, chat_input, uuid_state], [chatbot, chat_input]
    ).then(lambda: gr.Textbox(interactive=True), None, [chat_input])

    doc_input.upload(
        add_document, [doc_input, doc_list], [doc_list, file_table.dataset]
    )
    clear_doc_button.click(clear_document_list, outputs=[doc_list, file_table.dataset])


if __name__ == "__main__":
    demo.launch()
