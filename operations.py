from pysondb import db


class DatabaseOperator:
    def __init__(self):
        self.db = db.getDb('database.json')
    
    def add(self, data):
        self.db.add(data)
    
    def get(self):
        return self.db.get()[0]
 
    
"""
if __name__ == "__main__":
    dbjson =  DatabaseOperator()
    dbjson.add({"OrderNumber": "127", "Product": "Gucci Bag", "Delivery": "In Transit"})
    print(dbjson.get_all())
 """