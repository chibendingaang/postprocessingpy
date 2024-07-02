Here is the complete C++ code:

```cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <sstream>
#include <cstring>

int L, dtsymb, epss, choice, t_smoothness;
double dt, Lambda, Mu, epsilon;
int begin, end, fine_res, steps, stepcount;
std::string dtstr, epstr, paramstm, path, alphastr;

std::vector<std::vector<std::vector<double>>> Sp_a, Spnbr_a, E_loc, Sp_ngva, Spnbr_ngva, Eeta_loc, mconsv;
std::vector<std::vector<double>> Cxt, CExt;
std::vector<double> eta_;

double alpha;

void obtaincorrxt(int file_j, std::string path);
std::vector<std::vector<double>> calc_Cxt(std::vector<std::vector<std::vector<double>>> &Cxt_, int steps, std::vector<std::vector<std::vector<double>>> &spin);
void save_npy(const std::string &filename, std::vector<std::vector<double>> &array);

int main(int argc, char *argv[]) {
    if (argc < 10) {
        std::cerr << "Usage: " << argv[0] << " <L> <dtsymb> <Lambda> <Mu> <begin> <end> <epss> <choice> <t_smoothness>" << std::endl;
        return 1;
    }

    L = std::atoi(argv[1]);
    dtsymb = std::atoi(argv[2]);
    dt = 0.001 * dtsymb;

    Lambda = std::atof(argv[3]);
    Mu = std::atof(argv[4]);
    begin = std::atoi(argv[5]);
    end = std::atoi(argv[6]);

    epss = std::atoi(argv[7]);
    epsilon = std::pow(10, -epss);
    choice = std::atoi(argv[8]);

    t_smoothness = std::atoi(argv[9]);

    if (dtsymb == 2) {
        fine_res = 1 * t_smoothness;
    } else if (dtsymb == 1) {
        fine_res = 2 * t_smoothness;
    }

    dtstr = std::to_string(dtsymb) + "emin3";
    epstr = std::to_string(epss) + "eps_min3";

    if (Lambda == 1 && Mu == 0) {
        paramstm = "hsbg";
    } else if (Lambda == 0 && Mu == 1) {
        paramstm = "drvn";
    } else {
        paramstm = "a2b0";
    }

    alpha = (Lambda - Mu) / (Lambda + Mu);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << alpha;
    alphastr = oss.str();

    if (choice == 0) {
        param = "xp" + paramstm;
        if (paramstm != "a2b0") {
            path = "./" + param + "/L" + std::to_string(L) + "/2emin3";
        } else {
            path = "./" + param + "/L" + std::to_string(L) + "/alpha_" + alphastr + "/2emin3";
        }
    } else if (choice == 1) {
        param = "qw" + paramstm;
        if (paramstm != "a2b0") {
            path = "./" + param + "/L" + std::to_string(L) + "/2emin3";
        } else {
            param = "qwa2b0";
            path = "./" + param + "/2emin3/alpha_" + alphastr;
        }
    } else if (choice == 2) {
        if (paramstm == "a2b0") {
            param = "qw" + paramstm;
            path = "./" + param + "/2emin3/alpha_" + alphastr;
        } else {
            param = paramstm;
            path = "./" + param + "/L" + std::to_string(L) + "/2emin3";
        }
    }

    auto start = std::clock();

    for (int conf = begin; conf < end; ++conf) {
        obtaincorrxt(conf, path);
        if (param == "xpa2b0" || param == "qwa2b0") {
            std::string filename = "./Cxt_series_storage/L" + std::to_string(L) + "/alpha_ne_pm1/Cxt_t_" + dtstr + "_jump" + std::to_string(fine_res) + "_" + epstr + "_" + param + "_" + alphastr + "_" + std::to_string(conf) + "to" + std::to_string(conf + 1) + "config.npy";
            save_npy(filename, Cxt);
        }
    }

    std::cout << "Processing time = " << (std::clock() - start) / (double)CLOCKS_PER_SEC << " seconds." << std::endl;

    return 0;
}

void obtaincorrxt(int file_j, std::string path) {
    std::ifstream infile(path + "/spin_a_" + std::to_string(file_j) + ".dat");
    if (!infile) {
        std::cerr << "Error opening file: " << path + "/spin_a_" + std::to_string(file_j) + ".dat" << std::endl;
        exit(1);
    }

    std::vector<double> Sp_aj;
    double value;
    while (infile >> value) {
        Sp_aj.push_back(value);
    }
    infile.close();

    steps = Sp_aj.size() / (3 * L);
    Sp_a.resize(steps, std::vector<std::vector<double>>(L, std::vector<double>(3)));

    for (int i = 0; i < steps; ++i) {
        for (int j = 0; j < L; ++j) {
            for (int k = 0; k < 3; ++k) {
                Sp_a[i][j][k] = Sp_aj[i * L * 3 + j * 3 + k];
            }
        }
    }

    stepcount = std::min(steps, 521);
    Sp_a.resize(stepcount);

    int r = static_cast<int>(1.0 / dt);
    Cxt.resize(stepcount / fine_res + 1, std::vector<double>(L));
    CExt.resize(stepcount / fine_res + 1, std::vector<double>(L));
    Spnbr_a = Sp_a;

    for (auto &layer : Spnbr_a) {
        std::rotate(layer.begin(), layer.begin() + 1, layer.end());
    }

    E_loc.resize(stepcount, std::vector<std::vector<double>>(L, std::vector<double>(3)));
    for (int ti = 0; ti < stepcount; ++ti) {
        for (int x = 0; x < L; ++x) {
            for (int y = 0; y < 3; ++y) {
                E_loc[ti][x][y] = -Sp_a[ti][x][y] * Spnbr_a[ti][x][y];
            }
        }
    }

    std::vector<double> energ_1(stepcount);
    for (int i = 0; i < stepcount; ++i) {
        energ_1[i] = 0;
        for (int j = 0; j < L; ++j) {
            for (int k = 0; k < 3; ++k) {
                energ_1[i] += E_loc[i][j][k];
            }
        }
    }

    eta_.resize(stepcount * L * 3);
    for (int i = 0; i < stepcount * L * 3; ++i) {
        eta_[i] = (i % 6 < 3) ? 1 : -1;
    }

    eta_.resize(stepcount * L * 3);
    Sp_ngva = Sp_a;
    for (int i = 0; i < stepcount; ++i) {
        for (int j = 0; j < L; ++j) {
            for (int k = 0; k < 3; ++k) {
                Sp_ngva[i][j][k] *= eta_[i * L * 3 + j * 3 + k];
            }
        }
    }

    Spnbr_ngva = Spnbr_a;
    for (int i = 0; i < stepcount; ++i) {
        for (int j = 0; j < L; ++j) {
            for (int k = 0; k < 3; ++k) {
                Spnbr_ngva[i][j][k] *= eta_[i * L * 3 + j * 3 + k];
            }
        }
    }

    Eeta_loc.resize(stepcount, std::vector<std::vector<double>>(L, std::vector<double>(3)));
    for (int ti = 0; ti < stepcount; ++ti) {
        for (int x = 0; x < L; ++x) {
            for (int y = 0; y < 3; ++y) {
                Eeta_loc[ti][x][y] = -Sp_a[ti][x][y]
