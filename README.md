# ✈️ Aviation Cargo Summarizer (NLP)

### 📌 Project Overview
This project was developed for the **Eizen AI Machine Learning Assignment**. It utilizes the **FLAN-T5 (Base)** model to perform **Zero-Shot Inference**—summarizing complex technical text without specific fine-tuning.

### 🏢 Industry Use Case
Drawing from my **5 years of experience in Aviation Security**, I applied this model to summarize cargo screening protocols (Quikjet Cargo scenario). This demonstrates how LLMs can streamline compliance and safety reporting in high-stakes environments.

### 🛠️ Technical Stack
- **Model:** `google/flan-t5-base` (Seq2Seq Transformer)
- **Framework:** Hugging Face `transformers`
- **Method:** Zero-Shot Summarization with Prompt Templating
- **Tooling:** Developed using **Vibe Coding** principles in Cursor AI

### 🚀 How to Run
1. Clone the repo: https://github.com/gopi27-eng/Aviation-Cargo-Summarizer-LLM
2. Install requirements: `pip install -r requirements.txt`
3. Run the application: `python app.py`
   
### 📊 Sample Output
**Input Text:** "At Quikjet Cargo Airlines, all international shipments must undergo a dual-layer screening process..."

**Model Summary:** "Quikjet Cargo Airlines has a dual-layer screening process for international shipments."
