
# California House Valuer 🏠

A simple machine learning web app that predicts median house prices in California using demographic and housing data. Built with scikit-learn for regression modeling and Streamlit for an interactive user interface This project estimates California house prices using machine learning and Streamlit.

## 🔗 Live Demo
> 🔒 This app currently runs locally.  
> To try it out, follow the instructions in the *"How to Run Locally"* section below.

## 🧾 Project Structure
ml/
├── regression_model.py
├── app.py
├── model.pkl
├── README.md
├── requirements.txt


## ⚙️ How to Run Locally

1. Clone the repository  
```bash
git clone https://github.com/capybara-cpu/california-house-valuer.git
cd california-house-valuer
2. Install dependencies
```bash
Copy code
pip install -r requirements.txt
3. Run the app
```bash
streamlit run app.py

### 🏷️ Input Features Used

The model predicts median house prices using the following features:

| 🔢 Feature Name        | 📝 Description                                                                 |
|------------------------|--------------------------------------------------------------------------------|
| *Median Income*       | Average income of households in the area (in units of 10,000 USD)             |
| *House Age*           | Median age of the houses in the area                                          |
| *Average Rooms*       | Average number of rooms per dwelling (total rooms / households)               |
| *Average Bedrooms*    | Average number of bedrooms per dwelling (total bedrooms / households)         |
| *Population*          | Total population in the block                                                 |
| *Households*          | Number of households in the block                                             |
| *Latitude*            | Geographic latitude of the block (north-south position)                       |
| *Longitude*           | Geographic longitude of the block (east-west position)                        |

These features were used as inputs to train a *Linear Regression* model to estimate housing prices across California.


## 🛠 Tech Stack

- Python 🐍
- Streamlit 📊
- Scikit-Learn 🤖
- Pandas & NumPy 📈



