#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <string>
#include <memory>
#include <algorithm>
#include <fstream> 

//#include <highfive/highfive.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>


#include <omp.h>


using namespace HighFive;






struct GrayScottParameters {
    double Du;
    double Dv;
    double F; 
    double k; 
};

class GrayScottModel {
public:
    int width, height;
    std::vector<double> u, u_buffer;
    std::vector<double> v, v_buffer;
    GrayScottParameters params;

    GrayScottModel(int w, int h, GrayScottParameters p)
        : width(w), height(h), params(p)
    {
        u.resize(width * height, 1.0);
        v.resize(width * height, 0.0);
        u_buffer.resize(width * height, 0.0);
        v_buffer.resize(width * height, 0.0);

        int cx = width / 2, cy = height / 2;
        for (int j = cy - 10; j < cy + 10; ++j) {
            for (int i = cx - 10; i < cx + 10; ++i) {
                if(i >= 0 && i < width && j >= 0 && j < height) {
                    u[j * width + i] = 0.50;
                    v[j * width + i] = 0.25;
                }
            }
        }
    }

    inline double laplacian(const std::vector<double>& grid, int x, int y) {
        double sum = 0.0;
        int idx = y * width + x;
        sum -= 4 * grid[idx];
        if (y > 0)         sum += grid[(y - 1) * width + x];
        if (y < height - 1) sum += grid[(y + 1) * width + x];
        if (x > 0)         sum += grid[y * width + (x - 1)];
        if (x < width - 1)  sum += grid[y * width + (x + 1)];
        return sum;
    }

    void update(double dt, const int width, const int height) {

#pragma omp for schedule(guided)
        for (int x = 0 ; x < width ; x++) {
		for (int y = 0 ; y < height ; y++){
	            int idx = y * width + x;
	            double u_val = u[idx];
	            double v_val = v[idx];
	            double Lu = laplacian(u, x, y);
	            double Lv = laplacian(v, x, y);
	
	            double reaction = u_val * v_val * v_val;
	            u_buffer[idx] = u_val + (params.Du * Lu - reaction + params.F * (1 - u_val)) * dt;
	            v_buffer[idx] = v_val + (params.Dv * Lv + reaction - (params.F + params.k) * v_val) * dt;
	        }
	}
#pragma omp single
	{	
        std::swap(u, u_buffer);
        std::swap(v, v_buffer);
	}
    }
};


// Structure pour gérer les paramètres de sauvegarde
struct OutputSettings {
    bool enableOutput = true;    // Activer/Désactiver la sauvegarde
    int saveInterval = 100;      // Intervalle de sauvegarde
    std::string baseName;        // Nom de base des fichiers
};

// Fonction pour créer le contenu XDMF
std::string createXDMFContent(int width, int height, const std::string& hdf5Filename) {
    std::ostringstream xdmf;
    xdmf << "<?xml version=\"1.0\" ?>\n"
         << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n"
         << "<Xdmf Version=\"2.0\">\n"
         << " <Domain>\n"
         << "   <Grid Name=\"GrayScottModel\" GridType=\"Uniform\">\n"
         << "     <Topology TopologyType=\"2DCoRectMesh\" Dimensions=\"" << height << " " << width << "\"/>\n"
         << "     <Geometry GeometryType=\"ORIGIN_DXDY\">\n"
         << "       <DataItem Dimensions=\"2\" NumberType=\"Float\" Precision=\"8\" Format=\"XML\">0 0</DataItem>\n"
         << "       <DataItem Dimensions=\"2\" NumberType=\"Float\" Precision=\"8\" Format=\"XML\">" 
         << 1.0/width << " " << 1.0/height << "</DataItem>\n"
         << "     </Geometry>\n"
         << "     <Attribute Name=\"U\" AttributeType=\"Scalar\" Center=\"Node\">\n"
         << "       <DataItem Dimensions=\"" << height << " " << width << "\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">" 
         << hdf5Filename << ":/U</DataItem>\n"
         << "     </Attribute>\n"
         << "     <Attribute Name=\"V\" AttributeType=\"Scalar\" Center=\"Node\">\n"
         << "       <DataItem Dimensions=\"" << height << " " << width << "\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">" 
         << hdf5Filename << ":/V</DataItem>\n"
         << "     </Attribute>\n"
         << "   </Grid>\n"
         << " </Domain>\n"
         << "</Xdmf>\n";
    return xdmf.str();
}

// Initialisation du maillage initial
void saveInitialMesh(int width, int height, const std::string& hdf5Filename, const std::string& xdmfFilename) {
    std::ofstream xdmf(xdmfFilename, std::ios::out | std::ios::trunc);
    if (!xdmf.is_open()) {
        std::cerr << "Erreur: Impossible d'ouvrir " << xdmfFilename << std::endl;
        return;
    }
    xdmf << createXDMFContent(width, height, hdf5Filename);
    xdmf.close();
}

// Sauvegarde des données à une itération donnée
void saveIterationData(const std::string& hdf5Filename, int width, int height, 
                      const GrayScottModel& model, int iteration) {
    File file(hdf5Filename, File::ReadWrite | File::Create | File::Truncate);
    DataSpace space({static_cast<size_t>(height), static_cast<size_t>(width)});
    
    DataSet u_dataset = file.createDataSet<double>("U", space);
    DataSet v_dataset = file.createDataSet<double>("V", space);
    u_dataset.write_raw(model.u.data());
    v_dataset.write_raw(model.v.data());
    file.flush();
}

// Gestion principale de la sauvegarde
void handleOutput(int iteration, int width, int height, GrayScottModel& model, 
                 const OutputSettings& settings) {
    if (!settings.enableOutput) return;
    
    if (iteration == 0 || (iteration % settings.saveInterval == 0)) {
        std::string hdf5Filename = settings.baseName + "_iter" + std::to_string(iteration) + ".h5";
        std::string xdmfFilename = settings.baseName + "_iter" + std::to_string(iteration) + ".xdmf";
        
        saveIterationData(hdf5Filename, width, height, model, iteration);
        saveInitialMesh(width, height, hdf5Filename, xdmfFilename);
        
        std::cout << "Sauvegarde iteration " << iteration << ": " 
                  << hdf5Filename << ", " << xdmfFilename << std::endl;
    }
}


int main(int argc, char** argv) {

    const int width = 512;
    const int height = 512;

    // Configuration de la sortie
    OutputSettings settings;
    settings.enableOutput = false;    // Peut être mis à false pour performances
    settings.saveInterval = 1000;     // Sauvegarde toutes les 100 itérations
    settings.baseName = "grayscott";


    GrayScottParameters params = {0.16, 0.08, 0.060, 0.062};
    GrayScottModel model(width, height, params);

    // Sauvegarde initiale
    handleOutput(0, width, height, model, settings);

    const int steps = 2000;
    const double dt = 1.0;
    auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel
{
    for (int i = 0; i < steps; ++i) {
        model.update(dt, width, height);
#pragma omp single
{
        if (settings.enableOutput && (i % settings.saveInterval == 0)) {
	    handleOutput(i, width, height, model, settings);	
        }
        if (i % 1000 == 0) {
            std::cout << "Étape " << i << "\n";
        }
}
    }
}

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Simulation terminée en " << elapsed.count() << " secondes.\n";
    return 0;
}
