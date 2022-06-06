import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#expanding number is visable columns iof dataframe in console
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#importing instacart datasets
aisles = pd.read_csv("C:/Users/Nicole's PC/Desktop/Capstone Project/Instacart data/aisles.csv")
orders = pd.read_csv("C:/Users/Nicole's PC/Desktop/Capstone Project/Instacart data/orders.csv")
products = pd.read_csv("C:/Users/Nicole's PC/Desktop/Capstone Project/Instacart data/products.csv")
departments = pd.read_csv("C:/Users/Nicole's PC/Desktop/Capstone Project/Instacart data/departments.csv")
order_prior = pd.read_csv("C:/Users/Nicole's PC/Desktop/Capstone Project/Instacart data/order_products__prior.csv")
order_train = pd.read_csv("C:/Users/Nicole's PC/Desktop/Capstone Project/Instacart data/order_products__train.csv")

aisles.head()
aisles.info()
aisles.isnull().values.any()
aisles.hist(column = "aisle_id", grid = False)
aisles.describe()


orders.head()
orders.info()
orders.isnull().values.any()
orders["eval_set"].isnull().sum()
orders["order_number"].isnull().sum()
orders["order_dow"].isnull().sum()
orders["order_hour_of_day"].isnull().sum()
orders["days_since_prior_order"].isnull().sum()  #206209
orders.describe()
orders.eval_set.value_counts()

# graph of the different evaluation sets
y = sns.countplot(x = orders["eval_set"], data = orders)
for p in y.patches:
    y.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.25, p.get_height()+0.01))
y.set_xticklabels(y.get_xticklabels(), rotation = 90)
y.set(title = "Evaluation Set", xlabel = "Evaluation Set", ylabel = "frequency")


products.head()
products.info()
products.isnull().values.any()


departments.head()
departments.info()
departments.isnull().values.any()


order_prior.head()
order_prior.info()
order_prior.isnull().values.any()


order_train.head()
order_train.info()
order_train.isnull().values.any()

#merging the datasets into one
instacart = pd.merge(aisles, products, on = "aisle_id")
instacart = pd.merge(departments, instacart, on = "department_id")
instacart = pd.merge(order_prior, instacart, on = "product_id")
instacart = pd.merge(instacart, orders, on = "order_id")

#replacing NA in Days_since_prior_order to zero
instacart["days_since_prior_order"] = instacart["days_since_prior_order"].fillna(0)
instacart.tail()

instacart.isnull().values.any()
instacart["order_id"].count()
instacart["days_since_prior_order"].describe().apply("{0:.3f}".format)
instacart["order_dow"].describe().apply("{0:.3f}".format)
instacart["order_hour_of_day"].describe().apply("{0:.3f}".format)

#count the number of unique users
instacart["user_id"].nunique() #206209

#count the number of unique orders
instacart["order_id"].nunique() #3214874

instacart.eval_set.value_counts("train")


#creating graph of the number of orders placed by users
orders_per_customer = instacart.groupby("user_id")["order_number"].max().reset_index()
plt.figure(figsize=(25,15))
ax = sns.countplot(x = orders_per_customer["order_number"], data = instacart)
for p in ax.patches:
    ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.25, p.get_height()+0.01))
ax.set_title("Number of orders vs Number of Users makes these orders")
ax.set_xlabel("Numbewr of orders by User")
ax.set_ylabel("Number of Customers")
plt.savefig("Number of orders vs Number of Users makes these orders.png")


# Graph of number of reordered and not reordered products
reordered = sns.countplot(x = instacart["reordered"], data = instacart)
for p in reordered.patches:
    reordered.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.25, p.get_height()+0.01))
reordered.set_title("Reordered Products vs Not Reordered Products")
plt.savefig("Reordered")

#graph of day of the week orders were placed
dow = sns.countplot(x = instacart["order_dow"], data = instacart)
for p in dow.patches:
    dow.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.25, p.get_height()+0.01))
dow.set_title("Distribution of Purchases by Day of the Week")
dow.set_xlabel("Day of the Week")
dow.set_ylabel("Count")
plt.savefig("Distribution of Purchases by Day of the Week")

#graph of distribution of days since prior order
D_prior = sns.countplot(x = instacart["days_since_prior_order"], data = instacart)
D_prior.set_xticklabels(D_prior.get_xticklabels(), rotation = 90)
D_prior.set(title = "Distribution of User Orders by Days Since Last Order", xlabel = "Days Since Last Ordered", ylabel = "frequency")

# graph of hour of the day orders were made
e_set = sns.countplot(x = instacart["order_hour_of_day"], data = instacart)
e_set.set_xticklabels(e_set.get_xticklabels(), rotation = 90)
e_set.set(title = "Hour of the Day Orders were Made", xlabel = "Hours", ylabel = "frequency")

# Top 10 products ordered
items = sns.barplot(data = instacart.groupby('product_name')['add_to_cart_order'].sum().sort_values(ascending = False).reset_index()[0:10], x = 'product_name', y = 'add_to_cart_order')
items.set_xticklabels(items.get_xticklabels(), rotation = 90)
items.set(title = "Top 10 Products Ordered", xlabel = 'Product Name', ylabel = 'Number Ordered')

# Top 10 aisles ordered from
a = sns.barplot(data = instacart.groupby('aisle')['add_to_cart_order'].sum().sort_values(ascending = False).reset_index()[0:10], x = 'aisle', y = 'add_to_cart_order')
a.set_xticklabels(a.get_xticklabels(), rotation = 90)
a.set(title = "Top 10 Aisles Ordered From", xlabel = 'Aisle', ylabel = 'Units Ordered')

# Top 10 departments ordered from 
dept = sns.barplot(data = instacart.groupby('department')['add_to_cart_order'].sum().sort_values(ascending = False).reset_index()[0:10], x = 'department', y = 'add_to_cart_order')
dept.set_xticklabels(dept.get_xticklabels(), rotation = 90)
dept.set(title = "Top 10 Departments Ordered From", xlabel = 'Department', ylabel = 'Units Ordered')

#correlation matrix
corrmatrix = instacart.corr().round(2)
sns.heatmap(corrmatrix, annot = True)
