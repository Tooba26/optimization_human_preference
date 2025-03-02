from flask import Flask, request, render_template, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load fine-tuned model from Hugging Face
MODEL_NAME = "ToobiBabe/gpt_dpo"  # Replace with your actual model repo
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

@app.route('/')
def home():
    return render_template('index.html')

# ✅ Allow both POST (for chatbot) and GET (for browser testing)
@app.route('/generate', methods=['POST', 'GET'])
def generate():
    user_input = request.form.get('user_input') or request.args.get('user_input', '')

    if not user_input:
        return jsonify({'response': "Please enter a valid message."})

    # Tokenize input and generate response
    inputs = tokenizer(user_input, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=100,  
            num_return_sequences=1,  
            repetition_penalty=1.2,  
            temperature=0.7,  
            top_k=50,  
            top_p=0.9  
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ✅ Trim excessive repetition
    response_sentences = response.split('. ')  
    if len(response_sentences) > 2:
        response = '. '.join(response_sentences[:2]) + '.'  

    return jsonify({'response': response})  

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
