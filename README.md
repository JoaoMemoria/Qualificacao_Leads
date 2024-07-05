
# 📊 Lead Classification and Insights using Random Forest

This project demonstrates how to use a Random Forest Classifier to predict the likelihood of a lead converting into a sale based on various features. Additionally, it provides insights into the importance of different features in predicting sales.

## 🚀 Getting Started

### Prerequisites

Ensure you have the following Python packages installed:
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

You can install these packages using pip:
```bash
pip install pandas scikit-learn matplotlib seaborn
```

### Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/lead-classification.git
    cd lead-classification
    ```

2. Upload your leads CSV file to the directory or use the provided example leads file.

## 📂 Project Structure

```plaintext
├── leads_simulados.csv       # Example leads CSV file
├── lead_classification.ipynb # Jupyter Notebook with the implementation
└── README.md                 # This file
```

## 📋 CSV File Format

Ensure your CSV file follows this format:

| Nome        | Sobrenome | Email               | Telefone   | Cidade    | Estado | Empresa      | Cargo       | Fonte       | Data_Criacao | Venda_Feita |
|-------------|-----------|---------------------|------------|-----------|--------|--------------|-------------|-------------|--------------|-------------|
| João        | Silva     | joao@email.com      | 1234567890 | Fortaleza | CE     | Empresa X    | Analista    | Facebook    | 2023-01-01   | Sim         |
| Maria       | Oliveira  | maria@email.com     | 0987654321 | Recife    | PE     | Empresa Y    | Gerente     | Google Ads  | 2023-02-15   | Não         |
| ...         | ...       | ...                 | ...        | ...       | ...    | ...          | ...         | ...         | ...          | ...         |

## 🛠️ Usage

1. **Load Data from CSV:**
    ```python
    import pandas as pd

    # Load the data
    df = pd.read_csv('path/to/leads_simulados.csv')
    ```

2. **Pre-process the Data:**
    ```python
    # Convert the target column to binary
    df['Venda_Feita'] = df['Venda_Feita'].apply(lambda x: 1 if x == 'Sim' else 0)

    # Drop irrelevant columns
    df = df.drop(columns=['Email', 'Telefone'])

    # Convert categorical columns to numerical using get_dummies
    df = pd.get_dummies(df, columns=['Nome', 'Sobrenome', 'Cidade', 'Estado', 'Empresa', 'Cargo'], drop_first=True)
    ```

3. **Train the Model:**
    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    # Split the data into training and testing sets
    X = df.drop('Venda_Feita', axis=1)
    y = df['Venda_Feita']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    ```

4. **Evaluate the Model:**
    ```python
    from sklearn.metrics import classification_report, confusion_matrix

    # Make predictions
    y_pred = model.predict(X_test)

    # Print evaluation metrics
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    ```

5. **Feature Importance Insights:**
    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Get feature importances
    feature_importances = pd.DataFrame(model.feature_importances_,
                                       index=X_train.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)

    # Plot feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x=feature_importances.index, y=feature_importances['importance'])
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Importância', fontsize=14)
    plt.title('Importância das Features', fontsize=16)
    plt.tight_layout()
    plt.show()
    ```

## 📈 Results

After training and evaluating the model, you should see metrics indicating the performance of the model, such as precision, recall, and F1-score, as well as a confusion matrix and a plot of feature importances.

## 🤝 Contributing

Feel free to fork this repository and submit pull requests. Any enhancements or bug fixes are welcome!

## 📄 License

This project is licensed under the MIT License.

## 📧 Contact

For any questions or suggestions, feel free to contact [joaomemoria@gmail.com](mailto:joaomemoria@gmail.com).

---
