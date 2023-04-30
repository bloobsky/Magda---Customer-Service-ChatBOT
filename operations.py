from pysondb import db


class DatabaseOperator():

    def __init__(self, user):
        if(user == 'user'):
            self.db = db.getDb('database.json')
        else:
            self.db = db.getDb('navigation.json')
    
    def add(self, data):
        self.db.add(data)
    
    def get(self):
        return self.db.get()[0]
    
    def get_by_id(self):
        pass

    def get_all(self):
        return self.db.getAll()

    def delete(self, id):
        self.db.deleteById(id)

 
    
"""
if __name__ == "__main__":
    dbjson =  DatabaseOperator()
    dbjson.add({"OrderNumber": "127", "Product": "Gucci Bag", "Delivery": "In Transit"})
    print(dbjson.get_all())
"""