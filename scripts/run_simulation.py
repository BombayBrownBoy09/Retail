from models.retail_model import RetailModel

def main():
    params = {
        "example_param": 42
    }
    model = RetailModel(params)
    model.step()
    print("Simulation completed.")

if __name__ == "__main__":
    main()
