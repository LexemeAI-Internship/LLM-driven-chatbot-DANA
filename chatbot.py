import tiktoken
from openai.error import RateLimitError
import openai
import requests
import telebot
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from pydub import AudioSegment
from telegram import Message
from langchain import OpenAI, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.retrievers import SVMRetriever
from langchain.schema import Document
import numpy as np
from sklearn.cluster import KMeans
from langchain.chat_models import ChatOpenAI

state = 0
tmp = 0
document = 'none'
messages = [{"role": "system", "content": "You are an intelligent assistant."}]
BOT_USERNAME = ""  # Insert your bot username e.g. @mybot
BOT_TOKEN = ""  # Insert your bot token from botfather
bot = telebot.TeleBot(BOT_TOKEN)
openai_api_key = ""  # Insert your openai api key. Make sure you have enough credit
openai.api_key = openai_api_key
llm = OpenAI(openai_api_key=openai_api_key)
memory = ConversationBufferMemory(memory_key="chat_history")
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")


@bot.message_handler(commands=['start'])
def start(message):
    name = message.from_user.first_name
    bot.reply_to(message, rf"Hi {name}! How can I assist you?")


@bot.message_handler(commands=["help"])
def help(message):
    answer = (
        "Hi😀 You can do any of the following tasks with this AGPT bot:\n"
        "-Chat with our GPT-3.5-turbo bot\n-Send a TXT or PDF formatted file to summarize\n"
        "-Ask question about anything existed in you uploaded TXT or PDF formatted file")
    bot.reply_to(message, answer)


@bot.message_handler(commands=['summarize'])
def command_summarize(message: Message):
    global state
    state = 1
    bot.send_message(message.chat.id, "Send a TXT or PDF file that has less than 5 pages to summarize.")


@bot.message_handler(commands=['comprehensive_summarize_english'])
def command_comprehensive_summarize(message: Message):
    global state
    state = 2
    bot.send_message(message.chat.id, "Send a TXT or PDF file with more than 5 pages.\nYou can specify in "
                                      "the caption how many pages you want "
                                      "it summarized into."
                                      "\nExample:\n4 👉 summarize in 4 pages\n"
                                      "Note: If your file is in English, it shouldn't be more than 150 pages and "
                                      "if it's not in english, it shouldn't have more than 40 pages.\n"
                                      "Note: If your file have more pages, you can use the link below to convert your "
                                      "PDF file to TXT and then split it into 40-page sections. Now you can send each "
                                      "section to bot separately.\n"
                                      "https://www.zamzar.com/convert/pdf-to-txt/")


@bot.message_handler(commands=['comprehensive_summarize_persian'])
def command_comprehensive_summarize(message: Message):
    global state
    state = 4
    bot.send_message(message.chat.id, "یک فایل با فرمت PDF یا TXT ارسال کنید و در کپشن تعداد صفحاتی که میخواهید "
                                      "فایل خلاصه شود، مشخص کنید.\n"
                                      "مثال:\n 4 👈 در 4 صفحه خلاصه کن\n"
                                      "توجه: اگر فایل شما به زبان انگلیسی است، تعداد صفحات آن نباید بیش از 150 صفحه "
                                      "باشد و اگر فارسی است نباید بیش از 40 صفحه داشته باشد.\n\n"
                                      "توجه: اگر فایل شما تعداد صفحات بیشتری از محدودیت بالا دارد، میتوانید با استفاده "
                                      "از لینک زیر فایل PDF خود را به فایل TXT تبدیل کنید و سپس آن را به بخش های "
                                      "40-صفحه ای تقسیم کنید. حال میتوانید هر بخش را به صورت جداگانه برای ربات ارسال "
                                      "کنید.\n"
                                      "\nhttps://www.zamzar.com/convert/pdf-to-txt/")


@bot.message_handler(commands=['question_answer'])
def command_question_answer(message: Message):
    global state
    state = 3
    bot.send_message(message.chat.id,
                     "Please send a TXT or PDF file, and in the caption, ask a question related to the "
                     "content of the file.")


@bot.message_handler(func=lambda message: True, content_types=['text', 'voice'])
def echo(message: Message):
    global document, messages
    audio_types = ['audio/ogg', 'audio/mp4', 'audio/wav', 'audio/m4a', 'audio/mpeg3', 'audio/mp3']
    voice = message.voice
    chat_type = message.chat.type
    if chat_type in ['group', 'supergroup'] and voice not in audio_types:
        print("Group message received...")
        if BOT_USERNAME in str(message):
            message_tmp = str(message).replace(BOT_USERNAME, '').strip()
            messages.append({"role": "user", "content": str(message_tmp)}, )
            token_reset()
            chat = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k",
                                                messages=messages)
            reply = chat.choices[0].message.content
            messages.append({"role": "user", "content": reply}, )
            if len(reply) > 4095:
                bot.reply_to(message, reply[:4095])
                bot.reply_to(message, reply[4095:])
            else:
                bot.reply_to(message, reply)
            token_reset()
    else:
        if voice:
            if voice.mime_type in audio_types:
                print("Voice message received...")
                user_id = message.chat.id
                file_info = bot.get_file(message.voice.file_id)
                file = requests.get(f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_info.file_path}")
                with open("voice_message.ogg", "wb") as f:
                    f.write(file.content)
                # wav_format = AudioSegment.from_file("voice_message.ogg", format="ogg")
                # wav_format.export("voice_message.wav", format="wav")
                AudioSegment.from_ogg("voice_message.ogg").export("voice_message.wav", format="wav")
                voice_text = openai.Audio.transcribe("whisper-1", open("voice_message.wav"))
                messages.append({"role": "user", "content": voice_text['text']}, )
        elif message:
            print("Text message received...")
            messages.append({"role": "user", "content": str(message)}, )
        chat = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=messages)
        reply = chat.choices[0].message.content
        messages.append({"role": "user", "content": reply}, )
        if len(reply) > 4095:
            bot.reply_to(message, reply[:4095])
            bot.reply_to(message, reply[4095:])
        else:
            bot.reply_to(message, reply)
        token_reset()


def token_reset():
    global messages
    num_tokens = sum(len(encoding.encode(messages[i]["content"])) for i in range(1, len(messages)))
    print(num_tokens)
    if num_tokens >= 7000:
        messages = [messages[0]]
        print(messages)
        print(num_tokens)


def summarize(file_path):
    global document
    if document == 'pdf':
        print("PDF document is summarizing...")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=512)
        split_docs = text_splitter.split_documents(docs)
        print(len(split_docs))
        model = load_summarize_chain(llm=llm, chain_type="map_reduce")
        output = model.run(split_docs)
        return output
    elif document == 'text':
        print("Text file is summarizing...")
        with open(file_path, 'rb') as f:
            content = f.read().decode('utf-8')

        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"],
                                                       chunk_size=4000,
                                                       chunk_overlap=512)
        docs = text_splitter.create_documents([content])
        print(len(docs))
        map_prompt_template = """
                              Write a summary of this chunk of text that includes the main points and any 
                              important details.
                              {text}
                              """

        map_prompt = PromptTemplate(template=map_prompt_template,
                                    input_variables=["text"])

        combine_prompt_template = """
                              Write a concise summary of the following text delimited by triple backquotes.
                              Return your response in bullet points which covers the key points of the text.
                              ```{text}```
                              BULLET POINT SUMMARY:
                              """
        combine_prompt = PromptTemplate(template=combine_prompt_template,
                                        input_variables=["text"])
        summary_chain = load_summarize_chain(
            llm=llm,
            chain_type='map_reduce',
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
        )
        output = summary_chain.run(docs)
        return output


def comprehensive_summarize(file_path, num_pages, result_file):
    global document, state
    text = ""
    if document == 'pdf':
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        for page in pages:
            text += page.page_content
    elif document == 'text':
        with open(file_path, 'rb') as f:
            text = f.read().decode('utf-8')

    num_tokens = llm.get_num_tokens(text)
    # print(int(pages))
    # print(tokens)
    print(f"This book/article has {num_tokens} tokens in it")
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=10000,
                                                   chunk_overlap=3000)
    docs = text_splitter.create_documents([text])
    num_documents = len(docs)

    print(f"File is split up into {num_documents} documents")

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectors = []
    try:
        vectors = embeddings.embed_documents([x.page_content for x in docs])
    except RateLimitError as e:
        print(RateLimitError)
        return -1

    # If there are 500 words on average in each page, on my experience each cluster create about 0.55 page, so
    # num_clusters should be int(pages / 0.55)
    if num_pages == 0:
        num_clusters = 4
    else:
        num_clusters = int(num_pages / 0.55)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)
    closest_indices = []

    # Loop through the number of clusters you have
    for i in range(num_clusters):
        # Get the list of distances from that particular cluster center
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)

        # Find the list position of the closest one (using argmin to find the smallest distance)
        closest_index = np.argmin(distances)

        # Append that position to your closest indices list
        closest_indices.append(closest_index)

    selected_indices = sorted(closest_indices)
    print(selected_indices)

    llm3 = ChatOpenAI(temperature=0,
                      openai_api_key=openai_api_key,
                      max_tokens=1000,
                      model='gpt-3.5-turbo-16k'
                      )
    # map_prompt = """
    # You will be given a single passage of a book. This section will be enclosed in triple backticks (```)
    # Your goal is to give a summary of this section so that a reader will have a full understanding of what happened.
    # Your response should be about""" + str(tokens) + """ and fully encompass what was said in the passage.
    #
    # ```{text}```
    # FULL SUMMARY:
    # """
    language = ""
    if state == 4:
        language = " in persian language"
    map_prompt = """
        You will be given a single passage of a book. This section will be enclosed in triple backticks (```)
        Your goal is to give a summary of this section""" + language + """ so that a reader will have a full 
        understanding of what happened.
        Your response should be at least three paragraphs and fully encompass what was said in the passage.

        ```{text}```
        FULL SUMMARY:
        """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    map_chain = load_summarize_chain(llm=llm3,
                                     chain_type="stuff",
                                     prompt=map_prompt_template)
    selected_docs = [docs[doc] for doc in selected_indices]
    # Make an empty list to hold your summaries
    summary_list = []

    # Loop through a range of the length of your selected docs
    # start = time.process_time()
    for i, doc in enumerate(selected_docs):
        # Go get a summary of the chunk
        chunk_summary = map_chain.run([doc])
        print(len(chunk_summary))

        # Append that summary to your list
        summary_list.append(chunk_summary)
        print(f"Summary #{i} (chunk #{selected_indices[i]}) - Preview: {chunk_summary[:250]} \n")
    # print(time.process_time() - start)

    summaries = "\n".join(summary_list)
    # Convert it back to a document
    summaries = Document(page_content=summaries)
    with open(result_file, "w") as f:
        f.write(summaries.page_content)
    print(f"\n\nComprehensive summary:\n{summaries.page_content}")
    print(f"Your total summary has {llm.get_num_tokens(summaries.page_content)} tokens")
    return 0


def q_and_answer(file_path, question):
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(model_name=model_name,
                               model_kwargs=model_kwargs,
                               encode_kwargs=encode_kwargs)
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"],
                                                   chunk_size=4000,
                                                   chunk_overlap=512)

    if document == 'pdf':
        print("Answering using pdf file...")
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        split_docs = text_splitter.split_documents(docs)
        svm_retriever = SVMRetriever.from_documents(split_docs, embeddings=hf)
        print(len(split_docs))

    else:
        print("Answering using text file")
        with open(file_path, 'rb') as f:
            docs = f.read().decode('utf-8')
        text_chunks = text_splitter.split_text(docs)
        svm_retriever = SVMRetriever.from_texts(text_chunks, embeddings=hf)
        print(len(text_chunks))

    question_prompt_template = """Answer the question delimited by triple backquotes as precise as possible using the
    provided context delimited by triple backquotes. If the answer is not contained in the context, say "answer not
    available in context".
     \n Context: ``` {context}? ```
     \n Question: ``` {question} ```
     \n Answer:   """
    question_prompt = PromptTemplate(template=question_prompt_template,
                                     input_variables=["context", "question"])
    # question_prompt= SystemMessagePromptTemplate.from_template(question_prompt_template)

    combine_prompt_template = """Given the extracted content and the question delimited by triple backquotes , create a 
    final answer. If the answer is not contained in the context, say "answer not available". \n\n
    Summaries: {summaries}?
    Question: ``` {question} ```
    Answer:
    """
    combine_prompt = PromptTemplate(template=combine_prompt_template,
                                    input_variables=["summaries", "question"])
    docs_selected = svm_retriever.get_relevant_documents(question)
    map_reduce_chain = load_qa_chain(llm=llm,
                                     chain_type="map_reduce",
                                     question_prompt=question_prompt,
                                     combine_prompt=combine_prompt,
                                     return_intermediate_steps=True)
    outputs = map_reduce_chain({
        "input_documents": docs_selected,
        "question": question
    })
    return outputs['output_text']


@bot.message_handler(content_types=['document'])
def receive_file(message: Message):
    print("Document received...")
    global state
    global document
    path = None
    file = message.document
    caption = message.caption

    print(caption, type(caption))
    downloaded_file = None
    if file.mime_type == 'application/pdf':
        print("PDF file received...")
        document = 'pdf'
    if file.mime_type == 'text/plain':
        print("TXT file received...")
        document = 'text'
    if file.mime_type == 'text/plain' or file.mime_type == 'application/pdf':
        file_info = bot.get_file(file.file_id)
        file_name = file_info.file_path.split("/")[-1]
        downloaded_file = bot.download_file(file_info.file_path)
        path = fr'E:\ML Engineering and Data Science\Asr Gooyesh Pardaz(AGP)\Results\{file.file_name}'
        result_file_name = f"{file.file_name.split('.')[0]}_comprehensive_summary.txt"
        with open(path, 'wb') as f:
            f.write(downloaded_file)

        if state == 1:
            bot.reply_to(message, summarize(path))

        elif state == 2 or state == 4:
            if caption is None:
                caption = '0'
                bot.send_message(message.chat.id, "This may take up to 4 minutes")
            else:
                bot.send_message(message.chat.id, f"This may take up to "
                                                  f"{int(int(caption) / 0.55 * 25 / 60) + 2} minutes...")
            return_state = comprehensive_summarize(path, int(caption), result_file_name)
            if return_state == -1:
                bot.send_message(message.chat.id, "Your file has too many pages. Please send a smaller file!")
                return
            base_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"
            result_file = open(result_file_name, "rb")
            parameters = {
                "chat_id": message.chat.id,
                "caption": ""
            }
            files = {
                "document": result_file
            }

            bot.send_document(chat_id=message.chat.id,
                              document=open(result_file_name, "rb"), reply_to_message_id=message.message_id,
                              caption="Here is your summary file")

        elif state == 3:
            bot.reply_to(message, q_and_answer(path, caption))

    else:
        bot.reply_to(message, "Only TXT and PDF files are acceptable!")


print("Bot has been started")

bot.infinity_polling()
