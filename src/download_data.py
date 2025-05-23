import os
import urllib.request
from urllib.error import URLError, HTTPError

def download_mnist():
    os.makedirs("datasets", exist_ok=True)
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    dest = "datasets/mnist.npz"

    if os.path.exists(dest):
        print("Plik MNIST już istnieje: datasets/mnist.npz")
        return

    print("Pobieranie zbioru MNIST...")

    try:
        urllib.request.urlretrieve(url, dest)
        print("Zbiór danych został pobrany.")
    except HTTPError as e:
        print(f"Błąd HTTP: {e.code} – {e.reason}")
    except URLError as e:
        print(f"Błąd sieci: {e.reason}")
    except Exception as e:
        print(f"Nieoczekiwany błąd: {e}")
    finally:
        if not os.path.exists(dest):
            print("Pobieranie się nie powiodło. Możesz pobrać plik ręcznie:")
            print("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz")
            print("i zapisać go w katalogu: datasets/")

if __name__ == "__main__":
    download_mnist()
