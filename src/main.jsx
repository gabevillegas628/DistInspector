import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import HistogramTool from "./HistogramTool.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <HistogramTool />
  </StrictMode>
);
