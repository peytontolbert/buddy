from Nugget import NuggetAGI

def main():
    print("Starting up NuggetHR System")
    Nugget = NuggetAGI()
    while True:
        try:
            Nugget.run()

    
        except KeyboardInterrupt:
            print("Exiting...")
            break
            

if __name__ == "__main__":
    main()