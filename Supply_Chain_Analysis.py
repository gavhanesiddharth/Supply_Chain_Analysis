#!/usr/bin/env python
# coding: utf-8

# In[159]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[160]:


df = pd.read_csv("/Users/siddharth/Downloads/supply_chain_data.csv")
df


# In[161]:


df.info()


# In[162]:


df.isna().sum()


# In[163]:


df.drop_duplicates(inplace = True)
df


# In[164]:


plt.figure(figsize=(14, 12))
sns.heatmap(df.corr(numeric_only = True), annot=True, cmap = "coolwarm", vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()


# # Sales Analysis
# 

# In[165]:


Total_Revenue_Generated_per_product_type = df.groupby("Product type")["Revenue generated"].sum()
print(Total_Revenue_Generated_per_product_type)


# In[166]:


Top_10_products_sold = df.sort_values(by = "Number of products sold", ascending = False)
print(Top_10_products_sold.head(10))


# # Customer Analysis

# In[167]:


df["Customer demographics"].value_counts()


# In[168]:


sns.set_palette("GnBu")
sns.countplot(data = df, x = "Customer demographics")


# In[170]:


Avg_order_quantity_per_customer_segment = df.groupby("Customer demographics")["Order quantities"].mean().reset_index()
Avg_order_quantity_per_customer_segment["Order quantities"] = pd.to_numeric(Avg_order_quantity_per_customer_segment["Order quantities"])
Avg_order_quantity_per_customer_segment["Order quantities"] = Avg_order_quantity_per_customer_segment["Order quantities"].astype(int)
print(Avg_order_quantity_per_customer_segment)


# In[171]:


sns.set_context("notebook")
sns.set_style("white")
sns.barplot(data = Avg_order_quantity_per_customer_segment, x = "Customer demographics", y = "Order quantities")
plt.title("Average Order Quantites per Customer Demographics ")


# # Inventory Analysis

# In[173]:


Avg_stock_level_per_product_type = df.groupby("Product type")["Stock levels"].mean().reset_index()

print(Avg_stock_level_per_product_type)


# In[174]:


"""""
explode = (0.1, 0, 0)
plt.figure(figsize=(8, 6))  
plt.pie(Avg_stock_level_per_product_type["Stock levels"], explode=explode, labels=Avg_stock_level_per_product_type["Product type"], autopct='%1.1f%%', shadow = True)
plt.title('Average Stock Levels per Product Type')
plt.axis('equal')
"""""

#For average it is good to show bar plots instead of pie charts(percentages)

sns.barplot(data = Avg_stock_level_per_product_type, x = "Product type", y = "Stock levels")
plt.title('Average Stock Levels per Product Type')

for index, row in Avg_stock_level_per_product_type.iterrows():
    plt.text(index, row['Stock levels'] + 1, row['Stock levels'], ha='center', fontsize=10)


# In[175]:


Stockouts = df[df["Availability"] <= df["Order quantities"]].reset_index()
Stockouts


# # Shipping Analysis

# In[176]:


pd.set_option('display.max_columns', None)
print(df)


# In[177]:


Shipping_cost_across_carriers = df.groupby("Shipping carriers")["Shipping costs"].mean().reset_index()
Shipping_cost_across_carriers


# In[178]:


Transportation_lead_times = df.groupby("Transportation modes")["Lead time"].mean().reset_index()
Transportation_lead_times


# In[179]:


sns.boxplot(data = df, x = "Transportation modes", y = "Lead time")


# # Supplier Analysis

# In[180]:


Supplier_Performance = df.groupby("Supplier name")[["Lead time", "Defect rates"]].mean().reset_index()
Supplier_Performance.sort_values(by = ["Lead time", "Defect rates"], ascending = [True, True], inplace = True)


# In[181]:


sns.set_context("notebook")
fig, ax = plt.subplots(figsize=(14, 8))

sns.barplot(data=Supplier_Performance, x="Supplier name", y="Lead time", ax=ax, palette="BuGn", label="Lead time")

sns.barplot(data=Supplier_Performance, x="Supplier name", y="Defect rates", ax=ax, palette="RdPu", label="Defect rates", alpha=0.7)

ax.set_title("Lead Time and Defect Rates by Supplier")
ax.set_xticklabels(ax.get_xticklabels())
ax.set_xlabel("Supplier name")
ax.set_ylabel("Value")
ax.legend(title="Metric")

plt.tight_layout()


# In[182]:


Cost_effective_supplier = df.groupby("Supplier name")[["Manufacturing costs"]].mean().reset_index()
Cost_effective_supplier.sort_values(by = "Manufacturing costs", inplace = True)
Cost_effective_supplier


# In[183]:


sns.set_context("notebook")
sns.barplot(data = Cost_effective_supplier, x = "Supplier name", y = "Manufacturing costs")
plt.title("Manufacturing Cost by Supplier ")


# In[184]:


Most_effective_routes = df.groupby("Routes")["Costs"].mean().reset_index()
Most_effective_routes.sort_values(by = "Costs", inplace = True)
Most_effective_routes


# In[185]:


sns.set_context("notebook")
sns.barplot(data = Most_effective_routes, x = "Routes", y = "Costs")
plt.title("Average Cost by Routes")


# # Production Analysis

# In[186]:


Total_Production_Volume = df.groupby("Product type")["Production volumes"].sum().reset_index()
Total_Production_Volume


# In[187]:


sns.barplot(data = Total_Production_Volume, x = "Product type", y = "Production volumes")
plt.title("Total Production Volume by Product Type")


# In[188]:


# Manufacturing Lead Time
plt.figure(figsize=(10, 6))
sns.histplot(df['Manufacturing lead time'], bins=20, kde=True, color="green")
plt.title('Distribution of Manufacturing Lead Time')
plt.xlabel('Manufacturing Lead Time')
plt.ylabel('Frequency')
plt.show()


# In[189]:


plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='Supplier name', y='Manufacturing lead time')
plt.title('Manufacturing Lead Time by Supplier')
plt.xlabel('Supplier Name')
plt.ylabel('Manufacturing Lead Time')
plt.xticks()
plt.show()


# In[190]:


df.head()


# In[191]:


correlation_matrix = df[["Manufacturing lead time", "Price", "Production volumes", "Defect rates", "Manufacturing costs", "Order quantities"]].corr()
correlation_matrix


# In[192]:


sns.heatmap(correlation_matrix, annot = True, cmap = "coolwarm", vmin = -1, vmax = 1)


# Quality Control
# 
# Defect Rates: Calculate the defect rates for each product and identify any products with high defect rates.
# Inspection Results: Analyze the inspection results to identify common defects and areas for improvement.

# # Quality Control

# In[197]:


Defects_rate_per_product = df.groupby("Product type")["Defect rates"].mean()
Defects_rate_per_product


# In[206]:


Average_defect_rate = np.mean(df["Defect rates"])
Average_defect_rate


# In[221]:


defect_rate_by_products = df[["SKU", "Defect rates"]].sort_values(by = "Defect rates", ascending = False)
high_defect_products =defect_rate_by_products[defect_rate_by_products["Defect rates"] > Average_defect_rate]
high_defect_products


# In[227]:


high_defect_products["SKU"].size


# In[236]:


Inspection_results  = df["Inspection results"].value_counts()
print(Inspection_results)


# In[232]:


Areas_for_improvement = df[df["Inspection results"] == "Fail"]
Areas_for_improvement


# In[241]:


plt.figure(figsize=(8, 8))
explode = [0, 0.1, 0]
plt.pie(Inspection_results, labels=Inspection_results.index, autopct=lambda p: '{:.0f}'.format(p * sum(Inspection_results) / 100), shadow = True, explode = explode)
plt.title("Inspection Results Distribution")
plt.show()


# In[ ]:




