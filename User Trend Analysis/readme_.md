# User Trend Analysis

This project focuses on analyzing user trends from a dataset of food orders. The aim is to uncover insights and patterns in user behavior, including factors influencing order frequency, cost, delivery time, and the effectiveness of promotional codes. The analysis leverages statistical methods and data visualization to provide a comprehensive understanding of customer behavior in the context of online food delivery services.

## Project Objectives

The primary objectives of this project are:

- To identify key factors influencing customer orders.
- To understand the impact of promotions and discounts on user behavior.
- To analyze delivery times and their correlation with other factors.
- To provide actionable insights that can help in improving customer satisfaction and optimizing operational efficiency.

## Dataset

The dataset used in this analysis is imported from an Excel file (`Food_Orders.xlsx`) and includes the following features:

- **ID**: Unique identifier for each order.
- **Provider**: The service provider for the food order.
- **DateTime**: The timestamp of when the order was placed.
- **Distance**: The distance between the restaurant and the delivery location.
- **Status**: The status of the order (e.g., completed, canceled).
- **Cost**: The total cost of the order before any discounts or promotions.
- **PromoCode**: The promotional code applied to the order (if any).
- **Delivery Time**: The time taken to deliver the order.
- **Discount**: The discount amount applied to the order.
- **Paid Amount**: The final amount paid by the customer after applying discounts and promotional codes.
- **Delivery Charges**: The charges for delivery.
- **Surge Charges**: Additional charges during peak times.
- **Packaging Charges**: Charges for packaging the food items.

## Methodology

### 1. Data Cleaning and Preparation
- Imported the dataset and checked for proper data import.
- Cleaned the data by handling missing values, converting data types, and filtering out irrelevant information.

### 2. Exploratory Data Analysis (EDA)
- Performed descriptive statistics to summarize key data features.
- Used visualization techniques (e.g., histograms, box plots, scatter plots) to explore relationships between variables.
- Conducted trend analysis to observe changes over time or across different conditions.

### 3. Correlation and Statistical Analysis
- Generated correlation matrices to examine relationships between numerical variables.
- Performed statistical tests to identify significant differences in user behavior based on categorical variables.

### 4. Regression Analysis
- Built multiple regression models to predict outcomes such as delivery time and total cost based on factors like distance and surge charges.
- Evaluated model performance using metrics like R-squared and mean squared error (MSE).

### 5. Visualization and Reporting
- Visualized analysis results with charts and graphs.
- Generated a final report summarizing key insights, trends, and recommendations.

## Key Findings

- **Promotional Impact**: Promotional codes significantly influence the total cost and order frequency, highlighting their effectiveness in driving customer engagement.
- **Delivery Time Correlations**: Delivery times are strongly correlated with distance and surge charges, suggesting areas for service optimization.
- **Cost Analysis**: Higher distances and surge charges lead to increased total costs, even when discounts are applied.

## Conclusion

The User Trend Analysis project provides valuable insights into customer behavior in the online food delivery industry. By understanding these trends, businesses can make informed decisions to enhance services, optimize costs, and improve customer satisfaction. The statistical methods and models applied in this project offer a robust framework for analyzing similar datasets in other domains.

---

This README provides a detailed overview of the project, its objectives, methodology, and key findings, making it suitable for anyone interested in understanding the scope and results of the User Trend Analysis.
