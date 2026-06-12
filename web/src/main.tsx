import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { ThemeUIProvider } from "theme-ui";
import "./index.css";
import App from "./App.tsx";
import theme from "./theme.ts";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeUIProvider theme={theme}>
      <App />
    </ThemeUIProvider>
  </StrictMode>,
);
