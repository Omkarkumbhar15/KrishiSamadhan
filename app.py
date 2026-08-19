import streamlit as st
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# ================================
# CONFIG
# ================================
GROQ_API_KEY = "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"
FAISS_PATH = "data/vector_store/faiss_index.bin"
METADATA_PATH = "data/vector_store/metadata.json"

# ================================
# LOAD STATIC DATA
# ================================
@st.cache_resource
def load_data():
    index = faiss.read_index(FAISS_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

index, metadata = load_data()

@st.cache_resource
def load_embed_model():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

embed_model = load_embed_model()
# ⚠️ DO NOT CACHE
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
client = Groq(api_key=GROQ_API_KEY)

#modified
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Store user context
if "user_context" not in st.session_state:
    st.session_state.user_context = {
        "location": None,
        "crop": None
    }
# ================================
# SIDEBAR (FILTERS)
# ================================
st.sidebar.title("🌾 Settings")

domain_filter = st.sidebar.selectbox(
    "Select Domain",
    ["All", "crop", "soil", "pest", "weather"]
)

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.messages = []
    
st.sidebar.subheader("📍 Your Info")

location_input = st.sidebar.text_input("Enter your location")

if location_input:
    st.session_state.user_context["location"] = location_input

# ================================
# RETRIEVAL
# ================================
def retrieve(query, top_k=5):
    query_embedding = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, top_k * 2)

    results = []

    for i, idx in enumerate(indices[0]):
        item = metadata[idx]

        if domain_filter != "All" and item["domain"] != domain_filter:
            continue

        results.append({
            "text": item["text"],
            "source": item["source"],
            "domain": item["domain"],
            "score": float(distances[0][i])
        })

        if len(results) >= top_k:
            break

    return results

# ================================
# CONTEXT
# ================================
def build_context(chunks):
    return "\n".join([c["text"] for c in chunks])

# ================================
# CONFIDENCE SCORE
# ================================
def calculate_confidence(chunks, query):
    scores = [c["score"] for c in chunks]
    avg_score = np.mean(scores)

    # keyword match boost
    query_words = set(query.lower().split())
    match_count = 0

    for c in chunks:
        text_words = set(c["text"].lower().split())
        match_count += len(query_words & text_words)

    keyword_score = match_count / (len(query_words) + 1)

    final_score = (0.7 * avg_score) + (0.3 * keyword_score)

    return round(final_score * 100, 2)

# ================================
# PRODUCT RECOMMENDATION
# ================================
def get_products(query):
    query = query.lower()

    if "wheat" in query:
        return ["Urea Fertilizer", "DAP Fertilizer", "NPK 20-20-0"]
    elif "rice" in query:
        return ["Urea", "Potash", "Zinc Sulphate"]
    elif "pest" in query:
        return ["Neem Oil", "Chlorpyrifos", "Imidacloprid"]
    elif "soil" in query:
        return ["Vermicompost", "Organic Manure", "Gypsum"]
    else:
        return ["General NPK Fertilizer", "Organic Compost"]

# ================================
# RESOURCE LINKS
# ================================
def get_resources(domain):
    if domain == "crop":
        return ["https://agricoop.nic.in", "https://icar.org.in"]
    elif domain == "soil":
        return ["https://soilhealth.dac.gov.in"]
    elif domain == "pest":
        return ["https://ppqs.gov.in"]
    elif domain == "weather":
        return ["https://mausam.imd.gov.in"]
    else:
        return ["https://fao.org"]

def check_missing_info(query):
    missing = []

    if not st.session_state.user_context["location"]:
        missing.append("location")

    if not st.session_state.user_context["crop"]:
        if any(word in query.lower() for word in ["crop", "wheat", "rice"]):
            st.session_state.user_context["crop"] = query
        else:
            missing.append("crop")

    return missing
# ================================
# GENERATE ANSWER
# ================================
def generate_answer(query):
    chunks = retrieve(query)

    if not chunks:
        return "No data found.", [], 0, [], []
    # Enhance query with user context
    extra_context = ""

    if st.session_state.user_context["location"]:
        extra_context += f" Location: {st.session_state.user_context['location']}."

    if st.session_state.user_context["crop"]:
        extra_context += f" Crop: {st.session_state.user_context['crop']}."

    query = query + extra_context

    context = build_context(chunks)
    

    prompt = f"""
You are KrishiSamadhan, an agricultural assistant.

- Answer ONLY using context
- Give simple practical advice

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500
)

    answer = response.choices[0].message.content

    confidence = calculate_confidence(chunks, query)
    sources = list(set([c["source"] for c in chunks]))
    domain = chunks[0]["domain"]

    products = get_products(query)
    resources = get_resources(domain)

    return answer, sources, confidence, products, resources

# ================================
# UI
# ================================
st.title("🌾 KrishiSamadhan")
st.subheader("AI Assistant for Farmers")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
query = st.chat_input("Ask your farming question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    # 🔍 Check missing info
    missing = check_missing_info(query)

    if missing:
        follow_up = "🌱 To give you better and more accurate advice, please tell me your: " + ", ".join(missing)

        with st.chat_message("assistant"):
            st.info(follow_up)

        st.session_state.messages.append({
            "role": "assistant",
            "content": follow_up
        })

    else:
        with st.chat_message("assistant"):
            with st.spinner("🌱 Thinking..."):

                answer, sources, confidence, products, resources = generate_answer(query)

                st.markdown(answer)
                if "⚠️" in answer:
                    st.warning("Low confidence answer — consider refining your query or adding more details (crop, location).")
                
                if confidence > 70:
                    st.success(f"High Confidence: {confidence}%")
                elif confidence > 40:
                    st.info(f"Moderate Confidence: {confidence}%")
                else:
                    st.warning(f"Low Confidence: {confidence}%")

                with st.expander("📚 Sources"):
                    for s in sources:
                        st.write(f"- {s}")

                with st.expander("🔗 Learn More"):
                    for r in resources:
                        st.write(r)

                with st.expander("🛒 Recommended Products"):
                    for p in products:
                        st.markdown(f"✅ **{p}** – commonly used for better yield")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })