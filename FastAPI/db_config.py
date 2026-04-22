import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, Numeric
from sqlalchemy.orm import sessionmaker, declarative_base

# 1. Load environment variables from the .env file
load_dotenv()

# 2. Get the database URL
DATABASE_URL = os.getenv("MYSQL_URL")

# 3. Set up the SQLAlchemy Engine and Session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 4. Map the existing database table to a Python class
Base = declarative_base()

class IrisRecord(Base):
    __tablename__ = "iris_model"

    id = Column(Integer, primary_key=True, index=True)
    # We use Numeric(4,2) to perfectly match your DECIMAL(4,2) in SQL
    sepal_len = Column(Numeric(4, 2))
    sepal_width = Column(Numeric(4, 2))
    petal_len = Column(Numeric(4, 2))
    petal_width = Column(Numeric(4, 2))
    output = Column(Integer)

# 5. The function to insert a new prediction into the database
def save_prediction(sl, sw, pl, pw, predicted_class):
    db = SessionLocal() # Open a database connection
    try:
        # Create a new row object
        new_entry = IrisRecord(
            sepal_len=sl,
            sepal_width=sw,
            petal_len=pl,
            petal_width=pw,
            output=predicted_class
        )
        
        db.add(new_entry)  # Add it to the session
        db.commit()        # Save it to the database
        
        return True
    except Exception as e:
        db.rollback()      # Undo the transaction if something fails
        print(f"Failed to insert into database: {e}")
        return False
    finally:
        db.close()         # ALWAYS close the session to prevent memory leaks