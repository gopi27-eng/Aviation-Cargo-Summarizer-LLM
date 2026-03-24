import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def main():
    print("Loading FLAN-T5 Model and Tokenizer... (This may take a moment)")
    
    # Direct loading of the model and tokenizer
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Your Aviation Data (Step 4 of Assignment)
    aviation_text = """
    At Quikjet Cargo Airlines, all international shipments must undergo a dual-layer screening process. 
    First, the manifest is cross-referenced with the No-Fly list. Second, the physical 
    cargo is scanned using high-resolution X-ray technology to detect organic 
    and inorganic threats. Any shipment flagged during this process is moved 
    to a secure isolation zone for manual explosive trace detection (ETD) testing.
    """

    print("\n--- Original Aviation Text ---")
    print(aviation_text.strip())

    # Step 3: Zero-Shot Inference with Prompt Template
    # We explicitly add the instruction 'summarize: ' to the input
    input_text = f"summarize: {aviation_text}"
    
    # Convert text to numbers (tokens)
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate the summary
    print("\nGenerating summary...")
    outputs = model.generate(
        **inputs, 
        max_length=50, 
        min_length=15, 
        length_penalty=2.0, 
        num_beams=4, 
        early_stopping=True
    )

    # Convert numbers back to text
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n--- AI Generated Summary ---")
    print(summary)

if __name__ == "__main__":
    main()