import { Navigate, Route, Routes } from "react-router-dom";
import { PanelBoundary } from "./components/PanelBoundary";
import { Landing } from "./pages/Landing";
import { ArcStudio } from "./pages/ArcStudio";
import { Studio } from "./pages/Studio";
import { TourContext, useTour } from "./lib/tour";
import { AuthContext, useDemoAuth } from "./lib/auth";

/**
 * Each route is wrapped so a fault shows a message instead of a blank document.
 *
 * "/studio" is the SatQuery ARC Search & Review workspace. The original
 * claim-investigator Studio is kept at "/investigator" for reference.
 */
export default function App() {
  const tour = useTour();
  const auth = useDemoAuth();

  return (
    <AuthContext.Provider value={auth}>
      <TourContext.Provider value={tour}>
        <Routes>
          <Route
            path="/"
            element={
              <PanelBoundary panel="The landing page">
                <Landing />
              </PanelBoundary>
            }
          />
          <Route
            path="/studio"
            element={
              <PanelBoundary panel="The Search & Review workspace">
                <ArcStudio />
              </PanelBoundary>
            }
          />
          <Route
            path="/investigator"
            element={
              <PanelBoundary panel="The investigator studio">
                <Studio />
              </PanelBoundary>
            }
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </TourContext.Provider>
    </AuthContext.Provider>
  );
}
