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


struct IndicesSOA {
    std::vector<int> xs;
    std::vector<int> ys;
};



template<int Order>
struct StencilTraits;

// Spécialisation pour le stencil à 5 points
template<>
struct StencilTraits<5> {
    static constexpr int ghost_size = 1;
};

// Spécialisation pour le stencil à 9 points
template<>
struct StencilTraits<9> {
    static constexpr int ghost_size = 2;
};



class SpaceFillingCurve {
public:
    virtual IndicesSOA generate(int width, int height, int ghost_size) = 0;
    virtual ~SpaceFillingCurve() {}
};

/**
class ZOrderCurve : public SpaceFillingCurve {
private:
    inline int part1by1(int n) {
        n = (n | (n << 8)) & 0x00FF00FF;
        n = (n | (n << 4)) & 0x0F0F0F0F;
        n = (n | (n << 2)) & 0x33333333;
        n = (n | (n << 1)) & 0x55555555;
        return n;
    }
    inline int morton(int x, int y) {
        return (part1by1(y) << 1) | part1by1(x);
    }
public:
    std::vector<std::pair<int,int>> generate(int width, int height, int ghost_size) override {
        std::vector<std::pair<int,int>> indices;
        indices.reserve(width * height);
        for (int y = ghost_size; y < height; ++y) {
            for (int x = ghost_size; x < width; ++x) {
                indices.emplace_back(x, y);
            }
        }
        std::sort(indices.begin(), indices.end(), [&](const std::pair<int,int>& a, const std::pair<int,int>& b){
            return morton(a.first, a.second) < morton(b.first, b.second);
        });
        return indices;
    }
};
**/


class RowMajorCurve : public SpaceFillingCurve {
public:
    // Remarquez que la signature change pour retourner un IndicesSOA
    IndicesSOA generate(int width, int height, int ghost_size) override {
        IndicesSOA indices;
        // Réserver l'espace pour éviter des réallocations inutiles
        indices.xs.reserve(width * height);
        indices.ys.reserve(width * height);
        for (int y = ghost_size; y < height + ghost_size; ++y) {
            for (int x = ghost_size; x < width + ghost_size; ++x) {
                indices.xs.push_back(x);
                indices.ys.push_back(y);
            }
        }
        return indices;
    }
};


/**
class ColumnMajorCurve : public SpaceFillingCurve {
public:
    std::vector<std::pair<int,int>> generate(int width, int height, int ghost_size) override {
        std::vector<std::pair<int,int>> indices;
        for (int x = ghost_size; x < width; ++x) {
            for (int y = ghost_size; y < height; ++y) {
                indices.emplace_back(x, y);
            }
        }
        return indices;
    }
};
**/

/**
class HilbertCurve : public SpaceFillingCurve {
	// to do 
public:
    std::vector<std::pair<int,int>> generate(int width, int height) override {
        std::vector<std::pair<int,int>> indices;

        return indices;
    }
};
**/



template <int Order>
struct LaplacianStencil {
    inline static double compute(const std::vector<double>& grid, int x, int y, int width, int height) {
        constexpr int ghost_size = StencilTraits<Order>::ghost_size;
//	const int ghost_size = 1 ; 
	if constexpr(Order == 5){
	        double sum = 0.0;
		int width_with_ghost = width + 2*ghost_size ; 
	        int idx = y * (width_with_ghost) + x + ghost_size;
	        sum -= 4 * grid[idx];
	        double up     = grid[idx - (width_with_ghost)] * (y > ghost_size);
	        double down   = grid[idx + (width_with_ghost) ] * (y < height - 1 + 2 *ghost_size);
	        double left   = grid[idx - 1] * (x > ghost_size);
	        double right  = grid[idx + 1]  * (x < width_with_ghost -1);
		sum += up+down+left+right ; 
	        return sum;
	}
	else if constexpr( Order == 9){
	        int idx = y * width + x;
	        double center = grid[idx];
	        // Pour gérer les conditions aux limites, on peut utiliser la valeur centrale en bordure.
	        double north = (y > 0) ? grid[(y - 1) * width + x] : center;
	        double south = (y < height - 1) ? grid[(y + 1) * width + x] : center;
	        double west  = (x > 0) ? grid[y * width + (x - 1)] : center;
	        double east  = (x < width - 1) ? grid[y * width + (x + 1)] : center;
	        double nw    = (y > 0 && x > 0) ? grid[(y - 1) * width + (x - 1)] : center;
	        double ne    = (y > 0 && x < width - 1) ? grid[(y - 1) * width + (x + 1)] : center;
	        double sw    = (y < height - 1 && x > 0) ? grid[(y + 1) * width + (x - 1)] : center;
	        double se    = (y < height - 1 && x < width - 1) ? grid[(y + 1) * width + (x + 1)] : center;	
	        double lap = (4.0 * (north + south + east + west) +
	                      (nw + ne + sw + se) -
	                      20.0 * center) / 6.0;
	        return lap;
	}
    }
};


struct GrayScottParameters {
    double Du;
    double Dv;
    double F; 
    double k; 
};

template <int Order>
class GrayScottModel {
public:
    int width, height;
    std::vector<double> u, u_buffer;
    std::vector<double> v, v_buffer;
    GrayScottParameters params;
    static constexpr int ghost_size = StencilTraits<Order>::ghost_size; 

    GrayScottModel(int w, int h, GrayScottParameters p)
        : width(w), height(h), params(p)
    {
        u.resize((width+2*ghost_size) * (height+2*ghost_size), 1.0);
        v.resize((width+2*ghost_size) * (height+2*ghost_size), 0.0);
        u_buffer.resize((width+2*ghost_size) * (height+2*ghost_size), 0.0);
        v_buffer.resize((width+2*ghost_size) * (height+2*ghost_size), 0.0);

        int cx = width / 2, cy = height / 2;
        for (int j = cy - 10; j < cy + 10; ++j) {
            for (int i = cx - 10; i < cx + 10; ++i) {
                if(i >= 0 && i < width && j >= 0 && j < height) {
                    u[j * (width+2) + i] = 0.50;
                    v[j * (width+2) + i] = 0.25;
                }
            }
        }
    }


    inline double laplacian(const std::vector<double>& grid, int x, int y) {
        return LaplacianStencil<Order>::compute(grid, x, y, width, height); 
    }

    void update(double dt, const IndicesSOA& order) {

#pragma omp  for schedule(guided)
	for (int i = 0 ; i < order.xs.size(); i++){
//        for (const auto &coord : order) {
//	    int x = coord.first, y = coord.second;
            int x = order.xs[i], y = order.ys[i];
            int idx = y * (width+2) + x+1;
            double u_val = u[idx];
            double v_val = v[idx];
            double Lu = laplacian(u, x, y);
            double Lv = laplacian(v, x, y);

            double reaction = u_val * v_val * v_val;
            u_buffer[idx] = u_val + (params.Du * Lu - reaction + params.F * (1 - u_val)) * dt;
            v_buffer[idx] = v_val + (params.Dv * Lv + reaction - (params.F + params.k) * v_val) * dt;
        }
#pragma omp single
	{	
        std::swap(u, u_buffer);
        std::swap(v, v_buffer);
	}
    }


void zero_ghosts(std::vector<double>& grid, int width, int height, int ghost_size){
	int idx ; 
	// top and bottom ghosts
	for (int g = 0 ; g < ghost_size; g++){
		// bottom ghosts
		for (int i = 0 ; i < width + 2 * ghost_size ; i++){
			idx = g * width + i ; 
			grid[idx] = 0.0 ; 
		}
                // top ghosts
                for (int i = 0 ; i < width + 2 * ghost_size ; i++){
                        idx = (height + 2*ghost_size - g) * width - i ;
                        grid[idx] = 0.0 ;
                }
		
	}
	// left and right ghosts
	for (int j = ghost_size ; j < height + ghost_size; j++){
		// left ghosts
		for (int g = 0 ; g < ghost_size ; g++){
			idx = j * width + g ; 	
			grid[idx] = 0.0 ; 
		}		
                // right ghosts
                for (int g = 0 ; g < ghost_size ; g++){
                        idx = j * width + g + width + ghost_size;
                        grid[idx] = 0.0 ;
                }
		
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
template <int Order>
void saveIterationData(const std::string& hdf5Filename, int width, int height,
                      const GrayScottModel<Order>& model, int iteration) {
    File file(hdf5Filename, File::ReadWrite | File::Create | File::Truncate);
    DataSpace space({static_cast<size_t>(height), static_cast<size_t>(width)});

    DataSet u_dataset = file.createDataSet<double>("U", space);
    DataSet v_dataset = file.createDataSet<double>("V", space);
    u_dataset.write_raw(model.u.data());
    v_dataset.write_raw(model.v.data());
    file.flush();
}

// Gestion principale de la sauvegarde
template <int Order>
void handleOutput(int iteration, int width, int height, GrayScottModel<Order>& model,
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
//	std::string curveType = "column"
	std::string curveType = "row" ; 
//    std::string curveType = "classic"	
//  std::string curveType = "zorder";
//    std::string curveType = "hilbert";
    if (argc > 1) {
        curveType = argv[1];
    }
    std::unique_ptr<SpaceFillingCurve> curve;
    if (curveType == "hilbert") {
//        curve = std::make_unique<HilbertCurve>();
        std::cout << "Utilisation de la courbe Hilbert\n";
    } else if (curveType == "zorder") {
//        curve = std::make_unique<ZOrderCurve>();
        std::cout << "Utilisation de la courbe Z-order\n";
    } else if (curveType == "row") {
	    curve = std::make_unique<RowMajorCurve>();
    } else if (curveType == "column") {
//            curve = std::make_unique<ColumnMajorCurve>();
    }


    const int width = 512;
    const int height = 512;

	const int ghost_size = 1;

    // Configuration de la sortie
    OutputSettings settings;
    settings.enableOutput = true;    // Peut être mis à false pour performances
    settings.saveInterval = 100;     // Sauvegarde toutes les 100 itérations
    settings.baseName = "grayscott_5";



    auto order = curve->generate(width, height, ghost_size);

    
    GrayScottParameters params = {0.16, 0.08, 0.060, 0.062};
    GrayScottModel<5> model(width, height, params);

    // Sauvegarde initiale
    handleOutput(0, width+2*ghost_size, height+2*ghost_size, model, settings);


    const int steps = 5000;
    const double dt = 1.0;
    auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel
    {
    for (int i = 0; i < steps; ++i) {
        model.update(dt, order);
#pragma omp single
	{	
        if (settings.enableOutput && (i % settings.saveInterval == 0)) {
            handleOutput(i, width+2*ghost_size, height+2*ghost_size, model, settings);
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

