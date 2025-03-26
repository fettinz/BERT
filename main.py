import predict as p
import fit as f

if __name__ == "__main__":
    print("Premi 1 per addestrare il modello")
    print("Premi 2 per testare il modello sul dataset test")
    print("Premi 3 per inserire una frase da testare")
    choice = input()
    if choice == "1":
        f.train_model()
    elif choice == "2":
        p.load_model()
        p.test_model_on_test_set()
    elif choice == "3":
        p.load_model()
        print("Inserisci la frase da testare")
        sentence = input()
        p.test_sentence(sentence)