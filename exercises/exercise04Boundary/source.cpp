#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <algorithm>


class Exercise04Layer : public atcg::Layer
{
public:
    Exercise04Layer(const std::string& name) : atcg::Layer(name) {}

    Eigen::SparseMatrix<int> compute_boundary_operator(const std::shared_ptr<atcg::Mesh>& mesh)
    {
        /// Exercise: -Implement the \partial_2 boundary operator of the given mesh
        ///           -Define a fixed order for the edges
        ///           -You can use Eigen::SparseMatrix<int>::setFromTriplets to build the sparse matrix
        ///           -You can use eh.idx() to get a unique index for each edge (likewise for faces and vertices)

        // 

        // init sparse matrix
        int n_edges = mesh->n_edges();
        int n_faces = mesh->n_faces();
        Eigen::SparseMatrix<int> boundary_op(n_edges, n_faces);

        std::vector<Eigen::Triplet<int>> triplets; // (edge_index, face_index, sign)

        // iterate over all faces
        for (auto fh : mesh->faces())
        {
            int face_idx = fh.idx();

            // the face-halfedge circulator iterates over all halfedges of a face
            // in a consistent (counter-clockwise) order therefore defines a fixed orientation of the face
            for(auto heh = mesh->fh_ccwiter(fh); heh.is_valid(); ++heh) {
                // get the edge corresponding to this halfedge
                auto eh = mesh->edge_handle(*heh);
                int edge_idx = eh.idx();

                // determine the sign / orientation
                // choice: direction of the edge is given by its first halfedge
                auto heh_positive_orientation = mesh->halfedge_handle(eh, 0);

                // check if orientations match
                int sign = (*heh == heh_positive_orientation) ? 1 : -1;

                triplets.emplace_back(edge_idx, face_idx, sign);
            }
        }

        // construct sparse matrix
        boundary_op.setFromTriplets(triplets.begin(), triplets.end());

        return boundary_op;
    }

    Eigen::VectorXi compute_boundary_edges(const Eigen::SparseMatrix<int>& M)
    {
        /// Exercise: -Get a vector with n_edges() entries where non-boundary edges have value 0
        ///           -OpenMesh of course has methods to get boundary edges directly. DO NOT use those.

        // 

        // (n_edges x n_faces) * (n_faces x 1) = (n_edges x 1)
        Eigen::VectorXi c = Eigen::VectorXi::Ones(mesh->n_faces());
        Eigen::VectorXi boundary_edges = M * c;

        return boundary_edges;
    }

    void color_mesh(const std::shared_ptr<atcg::Mesh>& mesh, const Eigen::VectorXi& sum)
    {
        /// Exercise: -Color all vertices that make up the boundary edges red
        ///           -This framework does not support to color in edges directly, so you have to color in the
        ///           respective vertices -sum contains the output of 'compute_boundary_edges'.
        ///           - Again, DO NOT use OpenMesh functions

        // 

        for (int edge_idx = 0; edge_idx < sum.size(); ++edge_idx) 
        {
            if (sum[edge_idx] != 0)
            {
                // edge is boundary edge
                // get vertices of edge
                auto eh = mesh->edge_handle(edge_idx);
                auto heh = mesh->halfedge_handle(eh, 0);
                auto vh_from = mesh->from_vertex_handle(heh);
                auto vh_to   = mesh->to_vertex_handle(heh);
                // color vertices
                mesh->set_color(vh_from, atcg::Mesh::Color(255, 0, 0));
                mesh->set_color(vh_to, atcg::Mesh::Color(255, 0, 0));
            }
        }
    }

    // This is run at the start of the program
    virtual void onAttach() override
    {
        const auto& window = atcg::Application::get()->getWindow();
        float aspect_ratio = (float)window->getWidth() / (float)window->getHeight();
        camera_controller  = std::make_shared<atcg::CameraController>(aspect_ratio);

        mesh = atcg::IO::read_mesh("res/maxear.obj");
        mesh->request_vertex_colors();
        for(auto vt: mesh->vertices()) { mesh->set_color(vt, atcg::Mesh::Color(255)); }

        atcg::normalize(mesh);

        Eigen::SparseMatrix<int> M = compute_boundary_operator(mesh);
        Eigen::VectorXi sum        = compute_boundary_edges(M);
        color_mesh(mesh, sum);

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

        if(mesh && render_edges) atcg::Renderer::drawLines(mesh, glm::vec3(0), camera_controller->getCamera());
    }

    virtual void onImGuiRender() override
    {
        ImGui::BeginMainMenuBar();

        if(ImGui::BeginMenu("Rendering"))
        {
            ImGui::MenuItem("Show Render Settings", nullptr, &show_render_settings);
            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();

        if(show_render_settings)
        {
            ImGui::Begin("Settings", &show_render_settings);

            ImGui::Checkbox("Render Faces", &render_faces);
            ImGui::Checkbox("Render Edges", &render_edges);
            ImGui::Checkbox("Render Points", &render_points);

            ImGui::End();
        }
    }

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event& event) override { camera_controller->onEvent(event); }

private:
    std::shared_ptr<atcg::Mesh> mesh;
    std::shared_ptr<atcg::CameraController> camera_controller;

    bool show_render_settings = false;
    bool render_faces         = true;
    bool render_edges         = true;
    bool render_points        = false;
};

class Exercise04 : public atcg::Application
{
public:
    Exercise04() : atcg::Application() { pushLayer(new Exercise04Layer("Layer")); }

    ~Exercise04() {}
};

atcg::Application* atcg::createApplication()
{
    return new Exercise04;
}