# **Vector Database Documentation**

## **Overview**

The **Vector DB workflow** is a modular framework designed to assign categories to computer science questions.

---

## **System Architecture**

```
evaluation/
├── vectordb_workflow.py   # Central workflow orchestrator
├── chroma_db.py           # chroma_db implementation
├── database.py            # sql database implementation
├── models.py              # sql model initialization
```

Each component plays a distinct role in the evaluation lifecycle, ensuring modularity and reusability across different models and datasets.

---

## **1. Workflow Manager (`vectordb_workflow.py`)**

The **Workflow Manager** acts as the central control unit for the vector database.

### **Core Methods**
| Method | Description |
|--------|--------------|
| `load_embedding()` | Loads in the embedding model. |
| `load_data()` | Loads in the input data. |
| `get_category()` | Gets the category for a question id|
| `get_embedding()` | Converts the input question text into a word embedding|
| `assign_category()` | Assigns a category to an input question |
| `add()` | Takes in questions without categories as input, gets categories for each question, then calls populate |
| `populate_sql()` | Populates the SQL database |
| `populate_chroma()` | Populates the vector database |
| `populate()` | Takes in questions with categories as input, calls populate_sql and populate_chroma|

---

## **2. Vector Database (`chroma_db.py`)**

Sets up the chroma db vector database.

---

## **3. SQL Database (`database.py`)**

Sets up the SQL database.

---
## **4. SQL Models (`models.py`)**

Models for the SQL database.

---


