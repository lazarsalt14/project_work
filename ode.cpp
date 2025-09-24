#include <iostream>
#include <vector>
#include <complex>
#include <fstream>
#include <cmath>
#include <random>
#include <string>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <functional>
#include <future>
#include <array>

using cd = std::complex<double>;
using State = std::vector<cd>;

struct Data {
    double n_eff[4];
    double core_pitch;
    double ds;
    double twist_rate;
    double fiberLength;
    double wavelength;
    int avg_of;
};

// ---------------- Generate random phases ----------------
std::vector<double> random_phases(size_t N, std::mt19937 &g) {
    std::uniform_real_distribution<double> dist(-M_PI, M_PI);
    std::vector<double> th(N);
    for (size_t i = 0; i < N; ++i)
        th[i] = dist(g);
    return th;
}

// ---------------- Complex linear interpolation ----------------
cd interp1(const std::vector<double>& x, const std::vector<cd>& y, double xi) {
    if (xi <= x.front()) return y.front();
    if (xi >= x.back())  return y.back();
    size_t low = 0, high = x.size() - 1;
    while (high - low > 1) {
        size_t mid = (low + high) / 2;
        if (x[mid] > xi) high = mid; else low = mid;
    }
    double t = (xi - x[low]) / (x[high] - x[low]);
    return y[low]*(1.0 - t) + y[high]*t;
}

// ---------------- HetroODE ----------------
struct HetroODE {
    std::vector<double> betta;
    std::vector<double> z0;
    std::vector<cd> U0;
    double FL;
    std::vector<std::vector<double>> k;
    double twist_rate;
    double pitch;
    double R;

    HetroODE(const std::vector<double>& betta_,
             const std::vector<double>& z0_,
             const std::vector<cd>& U0_,
             double FL_,
             const std::vector<std::vector<double>>& k_,
             double twist_rate_, double pitch_, double R_)
        : betta(betta_), z0(z0_), U0(U0_), FL(FL_), k(k_),
          twist_rate(twist_rate_), pitch(pitch_), R(R_) {}

    State operator()(double z, const State& y) const {
        State dydt(4, cd(0,0));
        double theta0 = twist_rate * z;
        double D = pitch / std::sqrt(2.0);
        double phi[4];
        phi[0] = betta[0] * (z + (D / R) * (std::sin(theta0)/twist_rate));
        phi[1] = betta[1] * (z + (D / R) * ((std::cos(theta0)-1)/twist_rate));
        phi[2] = betta[2] * (z - (D / R) * (std::sin(theta0)/twist_rate));
        phi[3] = betta[3] * (z - (D / R) * ((std::cos(theta0)-1)/twist_rate));

        cd pert_cf = interp1(z0, U0, z);

        for (int tt=0; tt<4; ++tt)
            for (int pp=0; pp<4; ++pp)
                if(k[tt][pp] != 0.0)
                    dydt[tt] -= cd(0,1) * k[tt][pp] * y[pp] *
                                std::exp(cd(0,1)*(phi[tt]-phi[pp])) * pert_cf;

        return dydt;
    }
};

// ---------------- RK45 ----------------
struct RKResult {
    std::vector<double> x;
    std::vector<State> y;
};

RKResult rk45_dopri(const std::function<State(double,const State&)>& f,
                    double x0, double x1, State y0,
                    double rtol, double atol,
                    double h_init, double h_min, double h_max)
{
    // Butcher tableau
    static const double c2=0.2, c3=0.3, c4=0.8, c5=8.0/9.0;
    static const double a21=0.2;
    static const double a31=3.0/40.0, a32=9.0/40.0;
    static const double a41=44.0/45.0, a42=-56.0/15.0, a43=32.0/9.0;
    static const double a51=19372.0/6561.0, a52=-25360.0/2187.0,
                        a53=64448.0/6561.0, a54=-212.0/729.0;
    static const double a61=9017.0/3168.0, a62=-355.0/33.0,
                        a63=46732.0/5247.0, a64=49.0/176.0, a65=-5103.0/18656.0;
    static const double a71=35.0/384.0, a73=500.0/1113.0,
                        a74=125.0/192.0, a75=-2187.0/6784.0, a76=11.0/84.0;
    static const double e1=71.0/57600.0, e3=-71.0/16695.0, e4=71.0/1920.0,
                        e5=-17253.0/339200.0, e6=22.0/525.0, e7=-1.0/40.0;

    RKResult res;
    res.x.push_back(x0); res.y.push_back(y0);
    double h = h_init, x = x0;

    State k1(4), k2(4), k3(4), k4(4), k5(4), k6(4), k7(4), yt(4), y_new(4);

    while(x < x1){
        if(x+h > x1) h = x1 - x;

        k1 = f(x, y0);
        for(size_t i=0;i<4;++i) yt[i] = y0[i] + h*a21*k1[i];
        k2 = f(x+c2*h, yt);
        for(size_t i=0;i<4;++i) yt[i] = y0[i] + h*(a31*k1[i]+a32*k2[i]);
        k3 = f(x+c3*h, yt);
        for(size_t i=0;i<4;++i) yt[i] = y0[i] + h*(a41*k1[i]+a42*k2[i]+a43*k3[i]);
        k4 = f(x+c4*h, yt);
        for(size_t i=0;i<4;++i) yt[i] = y0[i] + h*(a51*k1[i]+a52*k2[i]+a53*k3[i]+a54*k4[i]);
        k5 = f(x+c5*h, yt);
        for(size_t i=0;i<4;++i) yt[i] = y0[i] + h*(a61*k1[i]+a62*k2[i]+a63*k3[i]+a64*k4[i]+a65*k5[i]);
        k6 = f(x+h, yt);

        for(size_t i=0;i<4;++i)
            y_new[i] = y0[i] + h*(a71*k1[i]+a73*k3[i]+a74*k4[i]+a75*k5[i]+a76*k6[i]);

        k7 = f(x+h, y_new);
        double err = 0.0;
        for(size_t i=0;i<4;++i){
            cd e = h*(e1*k1[i]+e3*k3[i]+e4*k4[i]+e5*k5[i]+e6*k6[i]+e7*k7[i]);
            double sc = atol + std::max(std::abs(y0[i]), std::abs(y_new[i]))*rtol;
            err += std::norm(e/sc);
        }
        err = std::sqrt(err/4.0);

        if(err <= 1.0){
            x += h;
            y0 = y_new;
            res.x.push_back(x);
            res.y.push_back(y0);
        }

        double fac = 0.9*std::pow(err,-0.2);
        fac = std::min(5.0, std::max(0.2, fac));
        h = std::min(h_max,std::max(h_min,h*fac));
    }

    return res;
}

// ---------------- run_single ----------------
std::vector<std::array<double,4>> run_single(const Data& data,
                                             double Bending_radius,
                                             int launch_power,
                                             const std::vector<std::vector<double>>& kappa,
                                             std::mt19937 &g,
                                             int run)
{
    double k0 = 2*M_PI/data.wavelength;
    std::vector<double> betta(4);
    for(int i=0;i<4;++i) betta[i] = k0*data.n_eff[i];

    double FL = data.fiberLength;
    double ds = data.ds;

    std::vector<double> z0;
    for(double z=0; z<=FL+1e-12; z+=ds) z0.push_back(z);

    // Complex random phases
    auto phases = random_phases(z0.size(), g);
    std::vector<cd> U0(z0.size());
    for(size_t i=0;i<z0.size(); ++i)
        U0[i] = std::exp(cd(0, phases[i])); // e^{j phi_i}

    State y0(4, cd(0,0));
    y0[launch_power-1] = cd(1.0,0.0);

    HetroODE ode(betta, z0, U0, FL, kappa, data.twist_rate, data.core_pitch, Bending_radius);

    size_t Nsteps = static_cast<size_t>(FL/ds);
    std::vector<std::array<double,4>> powers(Nsteps+1);
    for(auto &row : powers) row.fill(0.0);

    State y = y0;
    for(size_t i=0;i<=Nsteps;++i){
        for(int c=0;c<4;++c) powers[i][c] = std::norm(y[c]);
        if(i<Nsteps){
            auto sol = rk45_dopri([&](double zz,const State& yy){ return ode(zz,yy); },
                                   i*ds,(i+1)*ds,y,1e-6,1e-6,1e-3,1e-6, ds/2.0);
            y = sol.y.back();
        }
    }

    return powers;
}

// ---------------- Averaging wrapper ----------------
void core_power_hetro_avg(const Data& data, double Bending_radius,
                          int launch_power,
                          const std::vector<std::vector<double>>& kappa,
                          const std::string& out_prefix)
{
    size_t Nsteps = static_cast<size_t>(data.fiberLength/data.ds);
    std::vector<std::array<double,4>> avg(Nsteps+1);
    std::vector<std::array<double,4>> avg2(Nsteps+1); // sum of squares
    for(auto &row : avg) row.fill(0.0);
    for(auto &row : avg2) row.fill(0.0);

    std::vector<std::future<std::vector<std::array<double,4>>>> futures;
    for(int run=0; run<data.avg_of; ++run){
        futures.push_back(std::async(std::launch::async, [&,run]{
            std::mt19937 g(run+1);
            return run_single(data, Bending_radius, launch_power, kappa, g, run);
        }));
    }

    for(auto &fut : futures){
        auto powers = fut.get();
        for(size_t i=0;i<=Nsteps;++i)
            for(int c=0;c<4;++c) {
                avg[i][c] += powers[i][c];
                avg2[i][c] += powers[i][c] * powers[i][c];
            }
    }

    for(size_t i=0;i<=Nsteps;++i)
        for(int c=0;c<4;++c) avg[i][c] /= data.avg_of;


    std::vector<std::array<double,4>> stddev(Nsteps+1);
    for(size_t i=0;i<=Nsteps;++i)
        for(int c=0;c<4;++c)
            stddev[i][c] = (data.avg_of > 1) ?
                std::sqrt(avg2[i][c]/data.avg_of - avg[i][c]*avg[i][c]) : 0.0;


    std::ostringstream fname;
    fname << out_prefix << "_core" << launch_power << "_avg.csv";
    std::ofstream f(fname.str());
    f << "z,P_q1,P_q2,P_q3,P_q4,Std_q1,Std_q2,Std_q3,Std_q4\n";
    for(size_t i=0;i<=Nsteps;++i){
        double z = i*data.ds;
        f << z;
        for(int c=0;c<4;++c) f << "," << avg[i][c];
        for(int c=0;c<4;++c) f << "," << stddev[i][c];
        f << "\n";
    }
    std::cout << "Saved averaged result: " << fname.str() << "\n";
}

// ---------------- Main ----------------
int main(){
    Data data;
    data.n_eff[0] = 1.4464676;
    data.n_eff[1] = 1.4466484;
    data.n_eff[2] = 1.4466709;
    data.n_eff[3] = 1.4467458;
    data.core_pitch = 40e-6;
    data.ds = 0.1; 
    data.twist_rate = 0.1*M_PI;
    data.fiberLength = 10000.0;
    data.wavelength = 1550e-9;
    data.avg_of = 10; 

    double Bending_radius = 85e-3;

    std::vector<std::vector<double>> kappa = {
        {0, 0.0089484016, 1.32706e-05, 0.0086793744},
        {0.0090406079, 0, 0.0088845158, 1.27735e-05},
        {1.36451e-05, 0.0089420047, 0, 0.0086030505},
        {0.0089887085, 1.32556e-05, 0.0087662336, 0}
    };

    for(int power=1; power<=4; ++power){
        auto t_start = std::chrono::high_resolution_clock::now();
        std::cout << "Simulating (avg_of=" << data.avg_of << ") launch_power=" << power << "...\n";
        core_power_hetro_avg(data,Bending_radius,power,kappa,"Hetro_PL_core_cpp_dopri_opt_complex");
        auto t_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = t_end - t_start;
        std::cout << "Core " << power << " done in " << elapsed.count() << " s\n";
    }

    return 0;
}
