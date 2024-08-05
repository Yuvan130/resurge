import numpy as np
print(f"NumPy version: {np.__version__}")  # Print NumPy version to verify import
print(np.array([1, 2, 3]))
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
import json

app = Flask(__name__)
CORS(app)

# Authenticate with Hugging Face
login(token="Hugging face token")

# Load the Hugging Face model and tokenizer
model_name = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name, token="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", truncation=True, padding=True)
model = AutoModelForCausalLM.from_pretrained(model_name, token="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")

# Create a pipeline for text-generation
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.route('/generate_odoo_code', methods=['POST'])
def generate_odoo_code():
    try:
        # Get the requirements from the request
        inputReq = """
        Business Overview:

I own a mid-sized retail and wholesale business that operates both physical stores and an online e-commerce platform. We specialize in selling consumer electronics, home appliances, and related accessories. Our operations are spread across multiple locations, and we cater to both B2B and B2C markets.

Business Operations:

Retail and Wholesale: We have physical stores in several cities and a strong online presence through our e-commerce website. Our customers include both end consumers and businesses.

Inventory Management: We manage a large inventory with thousands of SKUs. This includes tracking stock levels across multiple warehouses, forecasting demand, and handling reorders.

Sales Channels: Our sales channels include in-store purchases, online orders, and B2B sales. We often deal with bulk orders from our wholesale clients and individual orders from retail customers.

Customer Relationship Management (CRM): We have a growing customer base, and we want to maintain strong relationships with them. This involves tracking customer interactions, managing leads, and running targeted marketing campaigns.

Accounting and Finance: We need to manage accounts payable, accounts receivable, general ledger, and financial reporting. Our accounting system must handle multiple currencies and comply with various tax regulations.

Human Resources: We employ a diverse workforce across multiple locations. We need an efficient system for managing employee records, payroll, performance evaluations, and recruitment.

E-commerce Platform: Our online store needs to be tightly integrated with our ERP system to ensure real-time inventory updates, smooth order processing, and efficient shipping management.

Supply Chain Management: We deal with multiple suppliers and need to manage procurement processes, monitor supplier performance, and optimize supply chain operations.

Requirements for Odoo ERP and E-commerce Solution:

Comprehensive ERP System: We need an all-in-one ERP solution that can manage inventory, sales, accounting, HR, and CRM. The system should be scalable to accommodate future growth and adaptable to our unique business processes.

Seamless Integration: The ERP system must seamlessly integrate with our e-commerce platform to ensure real-time data synchronization, particularly for inventory levels, order processing, and customer data.

Customizable Workflows: We require customizable workflows to align with our specific business processes, especially in sales, inventory management, and customer relationship management.

User-Friendly Interface: The solution should have a user-friendly interface that allows our staff to quickly learn and use the system effectively.

Reporting and Analytics: We need robust reporting and analytics tools to gain insights into our business operations, including sales performance, inventory turnover, financial health, and customer behavior.

Multi-location and Multi-currency Support: Since we operate across multiple locations and deal with international suppliers and customers, the system must support multi-location management and multi-currency transactions.

Mobile Accessibility: The ERP and e-commerce platform should be accessible via mobile devices to enable our team to manage operations on the go.

Customer Support: We expect ongoing support for system updates, troubleshooting, and user training to ensure the solution remains effective as our business evolves."""

        if not inputReq:
            inputReq = """
        Business Overview:

I own a mid-sized retail and wholesale business that operates both physical stores and an online e-commerce platform. We specialize in selling consumer electronics, home appliances, and related accessories. Our operations are spread across multiple locations, and we cater to both B2B and B2C markets.

Business Operations:

Retail and Wholesale: We have physical stores in several cities and a strong online presence through our e-commerce website. Our customers include both end consumers and businesses.

Inventory Management: We manage a large inventory with thousands of SKUs. This includes tracking stock levels across multiple warehouses, forecasting demand, and handling reorders.

Sales Channels: Our sales channels include in-store purchases, online orders, and B2B sales. We often deal with bulk orders from our wholesale clients and individual orders from retail customers.

Customer Relationship Management (CRM): We have a growing customer base, and we want to maintain strong relationships with them. This involves tracking customer interactions, managing leads, and running targeted marketing campaigns.

Accounting and Finance: We need to manage accounts payable, accounts receivable, general ledger, and financial reporting. Our accounting system must handle multiple currencies and comply with various tax regulations.

Human Resources: We employ a diverse workforce across multiple locations. We need an efficient system for managing employee records, payroll, performance evaluations, and recruitment.

E-commerce Platform: Our online store needs to be tightly integrated with our ERP system to ensure real-time inventory updates, smooth order processing, and efficient shipping management.

Supply Chain Management: We deal with multiple suppliers and need to manage procurement processes, monitor supplier performance, and optimize supply chain operations.

Requirements for Odoo ERP and E-commerce Solution:

Comprehensive ERP System: We need an all-in-one ERP solution that can manage inventory, sales, accounting, HR, and CRM. The system should be scalable to accommodate future growth and adaptable to our unique business processes.

Seamless Integration: The ERP system must seamlessly integrate with our e-commerce platform to ensure real-time data synchronization, particularly for inventory levels, order processing, and customer data.

Customizable Workflows: We require customizable workflows to align with our specific business processes, especially in sales, inventory management, and customer relationship management.

User-Friendly Interface: The solution should have a user-friendly interface that allows our staff to quickly learn and use the system effectively.

Reporting and Analytics: We need robust reporting and analytics tools to gain insights into our business operations, including sales performance, inventory turnover, financial health, and customer behavior.

Multi-location and Multi-currency Support: Since we operate across multiple locations and deal with international suppliers and customers, the system must support multi-location management and multi-currency transactions.

Mobile Accessibility: The ERP and e-commerce platform should be accessible via mobile devices to enable our team to manage operations on the go.

Customer Support: We expect ongoing support for system updates, troubleshooting, and user training to ensure the solution remains effective as our business evolves."""


        # Use pipeline to generate a response based on the input
        getRequirements = destructureRequest(inputReq)
        print("response", getRequirements)

        return jsonify({
            "status": "success",
            "generated_code": getRequirements
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

def destructureRequest(inputReq):
    prompt = f"""
Analyze the input and return the business and requirements as a JSON object. 
Input: {inputReq}
NOTE: Don't mention it is a JSON, just return it as JSON.
"""
    # Generate response using the pipeline
    response = pipe(prompt, max_length=1000, num_return_sequences=1)
    generated_text = response[0]['generated_text']
    
    # Extract the JSON part from the generated text
    try:
        json_start = generated_text.index('{')
        json_end = generated_text.rindex('}') + 1
        json_str = generated_text[json_start:json_end]
        return json.loads(json_str)
    except:
        return generated_text  # Return the full text if JSON parsing fails

def generate_code_from_requirements(requirements):
    prompt = f"""
Generate Odoo Python code for a module with the following requirements:

Module Name: {requirements.get('module_name', 'custom_module')}
Model Name: {requirements.get('model_name', 'custom.model')}
Fields: {requirements.get('fields', [])}

Include necessary imports, model definition, and field definitions.
"""
    # Generate code using the pipeline
    response = pipe(prompt, max_length=1000, num_return_sequences=1)
    generated_code = response[0]['generated_text']
    return generated_code

if __name__ == '__main__':
    app.run(debug=True)
