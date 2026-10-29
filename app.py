import requests
import streamlit as st

st.set_page_config(page_title='Grounded RAG', layout='wide')
st.title('Grounded RAG · document questions with citations')
base = st.sidebar.text_input('API URL', 'http://localhost:8000').rstrip('/')
file = st.file_uploader('Upload a text PDF or TXT', type=['pdf', 'txt'])
if file and st.button('Index document'):
    try:
        r = requests.post(f'{base}/documents', files={'file': (file.name, file.getvalue())}, timeout=180)
        r.raise_for_status()
        st.success(f"Indexed {r.json()['chunks']} chunks")
    except requests.RequestException as e:
        st.error(str(e))
question = st.text_input('Ask a question about your documents')
if st.button('Ask') and question:
    try:
        r = requests.post(f'{base}/ask', json={'question': question}, timeout=120)
        r.raise_for_status()
        result = r.json()
        st.markdown(result['answer'])
        for i, source in enumerate(result['sources'], 1):
            with st.expander(f"[{i}] {source['source']} · page {source['page']} · similarity {source['score']}"):
                st.write(source['text'])
    except requests.RequestException as e:
        st.error(str(e))
