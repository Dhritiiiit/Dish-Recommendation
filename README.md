# Dish-Recommendation
The Dish Recommendation System is a personal project designed to recommend food dishes based on user inputs—either a dish name or an image—while incorporating dietary preferences like vegetarian or spicy. It leverages multi-modal deep learning by processing images with ResNet and text with BERT, clusters dishes using HDBSCAN, and retrieves recommendations efficiently with FAISS. The system is deployed as an interactive web app using Streamlit, allowing users to explore personalized dish suggestions.
## 🚀 Objectives

- Recommend dishes based on either image or text input.
- Use **ResNet** for extracting image embeddings.
- Use **BERT** for extracting text (dish name) embeddings.
- Cluster dishes using **HDBSCAN** for better structure and interpretability.
- Retrieve recommendations via **FAISS** similarity search.
- Apply filters based on dietary preferences.
- Offer a user-friendly interface via **Streamlit**.

---

## 📥 Input

- Dish name or image.
- Optional: Dietary preferences (e.g., vegetarian, spicy, low-calorie).

## 📤 Output

- List of visually and semantically similar dishes.
- Filtered based on dietary restrictions/preferences.

---

## 🛠️ Technologies Used

- **Python**
- **PyTorch** – ResNet (image embeddings)
- **Hugging Face Transformers** – BERT (text embeddings)
- **FAISS** – Similarity search
- **HDBSCAN** – Clustering
- **Streamlit** – Web app interface
- **Pillow**, **NumPy**, **Pandas**

---

## 📊 Dataset

### Requirements
- Dish images
- Dish names
- Optional: dietary tags (vegetarian, spicy, etc.)

### Options
- **Food-101**: 101,000 images across 101 categories

