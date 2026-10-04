import { Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout.jsx";
import UniversalRestoration from "./pages/UniversalRestoration.jsx";
import HardRouted from "./pages/HardRouted.jsx";
import SoftMoE from "./pages/SoftMoE.jsx";
import FaceToSketch from "./pages/FaceToSketch.jsx";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Navigate to="/universal" replace />} />
        <Route path="/universal" element={<UniversalRestoration />} />
        <Route path="/hard-routed" element={<HardRouted />} />
        <Route path="/soft-moe" element={<SoftMoE />} />
        <Route path="/face-to-sketch" element={<FaceToSketch />} />
        <Route path="*" element={<Navigate to="/universal" replace />} />
      </Route>
    </Routes>
  );
}
