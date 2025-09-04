Code:
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import graphviz

# Load environment variables
load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-8b-8192"
)

# Streamlit UI setup
st.set_page_config(page_title="🌈 Rainbow", layout="centered")
st.title("🌈 Rainbow Formation")
st.caption("🔍 Ask the AI and visualize the explanation clearly!")

# Prompt input
prompt = st.text_input("💬 Enter a prompt:")

if st.button("Generate"):
    if not prompt.strip():
        prompt = "Explain how rainbows are formed"

    with st.spinner("🤖 Generating explanation..."):
        raw_response = llm.invoke(prompt)

    # ✅ Clean paragraph format (not split into bullets or scroll)
    st.success("✅ AI Explanation:")
    st.markdown(f"<div style='font-size:18px; line-height:1.7;'>{raw_response.content}</div>", unsafe_allow_html=True)

    # ✅ Flowchart
    if "rainbow" in prompt.lower():
        st.subheader("🌈 Rainbow Formation Flowchart")

        flowchart = graphviz.Digraph()
        flowchart.attr(rankdir='TB', size="10,12", fontsize="20")  # Top to bottom
        flowchart.attr('node', shape='box', fontsize='18', width='3', style='filled', fillcolor='lightblue')

        flowchart.node("1", "🌞 Sunlight enters a raindrop")
        flowchart.node("2", "↘ Light is refracted (bent)")
        flowchart.node("3", "🔁 Light reflects inside the raindrop")
        flowchart.node("4", "↗ Light exits and spreads out")
        flowchart.node("5", "🌈 A rainbow is formed!")

        flowchart.edges([("1", "2"), ("2", "3"), ("3", "4"), ("4", "5")])

        st.graphviz_chart(flowchart, use_container_width=True)

        st.info("💡 **Rainbows** are caused by sunlight interacting with water droplets through **refraction**, **internal reflection**, and **dispersion**.")
    else:
        st.info("💡 Flowchart will be shown only for rainbow-related prompts.")
      
