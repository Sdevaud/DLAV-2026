#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

void test_size(const vector<double>& liste1, const vector<double>& liste2);
double Loss(const vector<double>& echantillions, const vector<double>& resultats, double w);
double Gradient(vector<double>& echantillions, vector<double>& resultats, double w);
double new_w(double w, double grad, double mu);

int main() {
  // vector<double> echantilllions = {0.5, 1, 2, 3, 4, 5};
  // vector<double> resultats = {0.7, 2, 4, 6, 8, 12};
  vector<double> echantilllions = {1, 2, 3, 4};
  vector<double> resultats = {2, 4, 6, 8};
  test_size(echantilllions, resultats);

  double cst = 1e-10;
  double w = 1;
  size_t batch_size = 2;
  double mu = 0.04; // optimum takes 2 epochs
  double grad = 0;
  double loss = 0;
  size_t taille_sous_liste = echantilllions.size() / batch_size;
  size_t nbr_epochs = 0;
  size_t nbr_epochs_max = 100;


  do {
      for (size_t b = 0; b < batch_size; ++b) {
        vector<double> new_echantillions;
        vector<double> new_resultats;

        for(size_t j = 0; j < taille_sous_liste; ++j) {
          size_t idx = b * taille_sous_liste + j;
          new_echantillions.push_back(echantilllions[idx]);
          new_resultats.push_back(resultats[idx]);
        }

      loss = Loss(new_echantillions, new_resultats, w);
      grad = Gradient(new_echantillions, new_resultats, w);
      w = new_w(w, grad, mu);
    }
    ++nbr_epochs;
    cout << "w = " << w << endl;
    cout << "loss = " << loss << endl;
    cout << "nbr_epochs = " << nbr_epochs << endl;
    cout << "-----------------------------" << endl;
  } while (nbr_epochs < nbr_epochs_max && loss > cst);

  return 0;
}

double Loss(const vector<double>& echantillions, const vector<double>& resultats, double w) {
  test_size(echantillions, resultats);

  double loss = 0;
  for (size_t i = 0; i < echantillions.size(); ++i) {
    double y_pred = w * echantillions[i];
    loss += pow(y_pred - resultats[i], 2);
  }

  return loss / echantillions.size();
}

void test_size(const vector<double>& liste1, const vector<double>& liste2) {
  if (liste1.size() != liste2.size()) {
    cout << "error: size mismatch : X =" << liste1.size()
    << " Y =" << liste2.size() << endl;
    exit(1);
    return;
  }
  else return;
}

double Gradient(vector<double>& echantillions, vector<double>& resultats, double w) {
  test_size(echantillions, resultats);

  double grad = 0;
  for (size_t i = 0; i < echantillions.size(); ++i) {
    grad += 2 * (w * echantillions[i] - resultats[i]) * echantillions[i];
  }
  return grad / (double) echantillions.size();
}

double new_w(double w, double grad, double mu) {
  return w - mu * grad;
}