import pandas as pd


def make_fixture():
    orders = pd.DataFrame([
        [101, 10, 1, 0, 8, None, "prior"],
        [102, 10, 2, 1, 9, 2, "prior"],
        [103, 10, 3, 2, 10, 3, "prior"],
        [104, 10, 4, 3, 11, 4, "train"],
        [105, 10, 5, 4, 12, 5, "test"],
        [201, 20, 1, 0, 8, None, "prior"],
        [202, 20, 2, 1, 9, 2, "train"],
    ], columns=[
        "order_id", "user_id", "order_number", "order_dow",
        "order_hour_of_day", "days_since_prior_order", "eval_set",
    ])
    prior_products = pd.DataFrame([
        [101, 1, 1, 0],
        [101, 2, 2, 0],
        [102, 1, 1, 1],
        [102, 3, 2, 0],
        [103, 2, 1, 1],
        [103, 4, 2, 0],
        [201, 2, 1, 0],
    ], columns=["order_id", "product_id", "add_to_cart_order", "reordered"])
    train_products = pd.DataFrame([
        [104, 1, 1, 1],
        [104, 3, 2, 1],
        [202, 2, 1, 1],
    ], columns=["order_id", "product_id", "add_to_cart_order", "reordered"])
    products = pd.DataFrame([
        [1, "Apple", 10, 100],
        [2, "Bread", 10, 100],
        [3, "Milk", 11, 100],
        [4, "Pasta", 11, 101],
        [5, "Tea", 12, 101],
    ], columns=["product_id", "product_name", "aisle_id", "department_id"])
    aisles = pd.DataFrame([
        [10, "bakery"], [11, "dairy"], [12, "tea"],
    ], columns=["aisle_id", "aisle"])
    departments = pd.DataFrame([
        [100, "food"], [101, "pantry"],
    ], columns=["department_id", "department"])
    return orders, prior_products, train_products, products, aisles, departments
