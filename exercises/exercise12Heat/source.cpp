#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <algorithm>
#include <queue>

#include <numeric>

using VertexHandle             = atcg::Mesh::VertexHandle;
using EdgeHandle               = atcg::Mesh::EdgeHandle;
using HalfEdgeHandle           = atcg::Mesh::HalfedgeHandle;
using GeodesicDistanceProperty = OpenMesh::VPropHandleT<double>;

template<typename T>
struct LaplaceCotan
{
    T clampCotan(T v)
    {
        const T bound = 19.1;
        return (v < -bound ? -bound : (v > bound ? bound : v));
    }

    T triangleCotan(const atcg::TriMesh::Point& v0, const atcg::TriMesh::Point& v1, const atcg::TriMesh::Point& v2)
    {
        const auto d0   = v0 - v2;
        const auto d1   = v1 - v2;
        const auto d2   = v1 - v0;
        const auto area = atcg::areaFromMetric<T>(d0.norm(), d1.norm(), d2.norm());
        if(area > 1e-5) return clampCotan(d0.dot(d1) / area) / T(2.);
        return 1e-5;
    }

    atcg::Laplacian<T> calculate(const std::shared_ptr<atcg::Mesh>& mesh)
    {
        std::vector<Eigen::Triplet<T>> edge_weights;

        for(auto e_it = mesh->edges_begin(); e_it != mesh->edges_end(); ++e_it)
        {
            uint32_t i = e_it->v0().idx();
            uint32_t j = e_it->v1().idx();

            const auto h0 = e_it->h0();
            const auto h1 = e_it->h1();

            const auto p0 = h0.to();
            const auto p1 = h1.to();

            T weight = 0;
            if(!mesh->is_boundary(h0))
            {
                const auto p2 = h0.next().to();
                weight += triangleCotan(mesh->point(p0), mesh->point(p1), mesh->point(p2)) / T(2.);
            }

            if(!mesh->is_boundary(h1))
            {
                const auto p2 = h1.next().to();
                weight += triangleCotan(mesh->point(p0), mesh->point(p1), mesh->point(p2)) / T(2.);
            }

            edge_weights.emplace_back(i, j, weight);
            edge_weights.emplace_back(j, i, weight);
            edge_weights.emplace_back(i, i, -weight);
            edge_weights.emplace_back(j, j, -weight);
        }

        std::vector<Eigen::Triplet<T>> vertex_weights;

        for(auto v_it = mesh->vertices_begin(); v_it != mesh->vertices_end(); ++v_it)
        {
            T weight = mesh->area(*v_it) / T(3.);
            vertex_weights.emplace_back(v_it->idx(), v_it->idx(), weight);
        }

        size_t N = mesh->n_vertices();

        atcg::Laplacian<T> laplace;
        laplace.S.resize(N, N);
        laplace.M.resize(N, N);

        laplace.S.setFromTriplets(edge_weights.begin(), edge_weights.end());
        laplace.M.setFromTriplets(vertex_weights.begin(), vertex_weights.end());

        return laplace;
    }
};

class Exercise12Layer : public atcg::Layer
{
public:
    Exercise12Layer(const std::string& name) : atcg::Layer(name) {}

    void colorize_mesh(const std::shared_ptr<atcg::Mesh>& mesh, const OpenMesh::VPropHandleT<double>& vertexProperty)
    {
        assert(mesh->has_vertex_colors());
        double max_value = -std::numeric_limits<double>::infinity();
        double min_value = std::numeric_limits<double>::infinity();
        for(auto vh: mesh->vertices())
        {
            double value = mesh->property(vertexProperty, vh);
            if(!std::isfinite(value)) continue;
            max_value = std::max(max_value, value);
            min_value = std::min(min_value, value);
        }
        std::cout << min_value << " " << max_value << std::endl;
        for(auto vh: mesh->vertices())
        {
            double value = mesh->property(vertexProperty, vh);
            if(std::isfinite(value))
            {
                mesh->set_color(vh, {255 * (value - min_value) / (max_value - min_value), 0, 0});
            }
            else
            {
                mesh->set_color(vh, {0, 0, 255});
            }
        }
    }

    void cosine_colorize_mesh(const std::shared_ptr<atcg::Mesh>& mesh,
                              const OpenMesh::VPropHandleT<double>& vertexProperty,
                              const double periods)
    {
        assert(mesh->has_vertex_colors());
        double max_value = -std::numeric_limits<double>::infinity();
        double min_value = std::numeric_limits<double>::infinity();
        for(auto vh: mesh->vertices())
        {
            double value = mesh->property(vertexProperty, vh);
            if(!std::isfinite(value)) continue;
            max_value = std::max(max_value, value);
            min_value = std::min(min_value, value);
        }

        for(auto vh: mesh->vertices())
        {
            double value =
                ((mesh->property(vertexProperty, vh) - min_value) / (max_value - min_value)) * 2 * M_PI * periods;
            if(std::isfinite(value))
            {
                mesh->set_color(
                    vh,
                    {static_cast<unsigned char>(value / (2 * M_PI * periods) * 255 * (1.0 + cos(value)) / 2.0), 0, 0});
            }
            else
            {
                mesh->set_color(vh, {0, 0, 255});
            }
        }
    }

    void compute_matrices(const std::shared_ptr<atcg::Mesh>& mesh, double t)
    {
        /// Exercise: Prefactor the matrices A-tL_C and L_C
        ///           Use this->luAtLc and this->luLc for this
        LaplaceCotan<double> laplace_calculator;
        atcg::Laplacian<double> laplace  = laplace_calculator.calculate(mesh);
        Eigen::SparseMatrix<double> AtLc = laplace.M - t * laplace.S;
        Eigen::SparseMatrix<double> Lc   = laplace.S;
        this->luAtLc.compute(AtLc);
        this->luLc.compute(Lc);
    }

    void compute_heat_geodesics(const std::shared_ptr<atcg::Mesh>& mesh,
                                const std::vector<VertexHandle>& start_vhs,
                                GeodesicDistanceProperty& distance_property)
    {
        if(start_vhs.empty()) return;

        Eigen::VectorXd u0 = Eigen::VectorXd::Zero(mesh->n_vertices());
        for(auto start_vh: start_vhs) { u0[start_vh.idx()] = 1.0; }

        /// Exercise: Use the precomputed matrices to solve the heat equation for one time step
        ///           Store the result in an Eigen::VectorXd u

        Eigen::VectorXd u = this->luAtLc.solve(u0);

        std::vector<OpenMesh::Vec3d> face_grad_u(mesh->n_faces(), OpenMesh::Vec3d(0, 0, 0));
        for(auto fh: mesh->faces())
        {
            /// Exercise: Compute grad(u) for each triangle inside the mesh
            ///           Use mesh->opposite_halfedge_handle to get the half edge opposite to a given vertex
            const auto n = mesh->normal(fh);
            for(auto vh: fh.vertices())
            {
                auto heh  = mesh->opposite_halfedge_handle(fh, vh);
                auto v0   = mesh->from_vertex_handle(heh);
                auto v1   = mesh->to_vertex_handle(heh);
                auto p0   = mesh->point(v0);
                auto p1   = mesh->point(v1);
                auto step = n.cross(p1 - p0);
                face_grad_u[fh.idx()] -= u[vh.idx()] * step;    // Flipped sign
            }
            face_grad_u[fh.idx()].normalize();    // Normalized
        }

        Eigen::VectorXd vertex_div_u = Eigen::VectorXd::Zero(mesh->n_vertices());
        for(auto vh: mesh->vertices())
        {
            /// Exercise: Compute the divergence of X
            ///           Remember to handle boundary edges

            auto pi     = mesh->point(vh);
            double& div = vertex_div_u[vh.idx()];
            for(auto h_it = mesh->cvoh_ccwbegin(vh); h_it != mesh->cvoh_ccwend(vh); ++h_it)
            {
                // The edge and the next edge belong to a common face,
                // except when the current edge is a boundary edge. In that case
                // we skip it, as it will be the next_heh of another halfedge later.
                if(mesh->is_boundary(*h_it)) continue;
                
                auto fh = mesh->face_handle(*h_it);
                auto next_heh = mesh->next_halfedge_handle(*h_it);
                auto pj = mesh->point(mesh->to_vertex_handle(*h_it));
                auto pk = mesh->point(mesh->to_vertex_handle(next_heh));
                
                auto eij = pj - pi;
                auto eik = pk - pi;
                
                LaplaceCotan<double> laplace_builder;
                double cot_k = laplace_builder.triangleCotan(pi, pj, pk);
                double cot_j = laplace_builder.triangleCotan(pi, pk, pj);
                
                auto X = face_grad_u[fh.idx()];
                
                // Divergence formula
                div += 0.5 * (cot_j * (eik | X) + cot_k * (eij | X));
            }
        }

        /// Exercise: Solve for phi using the vertex divergence and the precomputed matrices Lc
        Eigen::VectorXd phi = luLc.solve(vertex_div_u);

        for(auto vh: mesh->vertices()) { mesh->property(distance_property, vh) = phi[vh.idx()]; }
    }

    // This is run at the start of the program
    virtual void onAttach() override
    {
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller  = std::make_shared<atcg::CameraController>(aspect_ratio);

        mesh = atcg::IO::read_mesh("res/bunny.obj");
        mesh->request_vertex_colors();

        mesh->add_property(distance_property);

        mesh->request_face_normals();
        mesh->update_face_normals();

        std::vector<VertexHandle> start_vhs;
        start_vhs.push_back(mesh->vertex_handle(start_id));
        double t = 0.1;

        compute_matrices(mesh, t);
        compute_heat_geodesics(mesh, start_vhs, distance_property);

        cosine_colorize_mesh(mesh, distance_property, 32);

        mesh->uploadData();
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        camera_controller->onUpdate(delta_time);

        atcg::Renderer::clear();

        if(mesh && render_faces)
            atcg::Renderer::draw(mesh, atcg::ShaderManager::getShader("base"), camera_controller->getCamera());

        if(mesh && render_points)
            atcg::Renderer::drawPoints(mesh,
                                       glm::vec3(0),
                                       atcg::ShaderManager::getShader("base"),
                                       camera_controller->getCamera());

        if(mesh && render_edges) atcg::Renderer::drawLines(mesh, glm::vec3(1), camera_controller->getCamera());
    }

    virtual void onImGuiRender() override
    {
        ImGui::BeginMainMenuBar();

        if(ImGui::BeginMenu("Rendering"))
        {
            ImGui::MenuItem("Show Render Settings", nullptr, &show_render_settings);
            ImGui::MenuItem("Show Geodesics Settings", nullptr, &show_geodesics_settings);

            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();

        if(show_render_settings)
        {
            ImGui::Begin("Settings", &show_render_settings);

            ImGui::Checkbox("Render Vertices", &render_points);
            ImGui::Checkbox("Render Edges", &render_edges);
            ImGui::Checkbox("Render Mesh", &render_faces);
            ImGui::End();
        }

        if(show_geodesics_settings)
        {
            ImGui::Begin("Geodesics");

            if(ImGui::SliderInt("Start vertex", &start_id, 0, mesh->n_vertices() - 1))
            {
                std::vector<VertexHandle> start_vhs;
                start_vhs.push_back(mesh->vertex_handle(start_id));

                compute_heat_geodesics(mesh, start_vhs, distance_property);

                cosine_colorize_mesh(mesh, distance_property, 32);
                mesh->uploadData();
            }

            ImGui::End();
        }
    }

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event& event) override
    {
        camera_controller->onEvent(event);

        atcg::EventDispatcher dispatcher(event);
    }

private:
    std::shared_ptr<atcg::CameraController> camera_controller;
    std::shared_ptr<atcg::Mesh> mesh;

    Eigen::SparseLU<Eigen::SparseMatrix<double>> luAtLc;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> luLc;

    int start_id = 3225;

    bool show_render_settings    = false;
    bool render_faces            = true;
    bool render_points           = false;
    bool render_edges            = false;
    bool show_geodesics_settings = true;
    GeodesicDistanceProperty distance_property;
};

class Exercise12 : public atcg::Application
{
public:
    Exercise12() : atcg::Application() { pushLayer(new Exercise12Layer("Layer")); }

    ~Exercise12() {}
};

atcg::Application* atcg::createApplication()
{
    return new Exercise12;
}