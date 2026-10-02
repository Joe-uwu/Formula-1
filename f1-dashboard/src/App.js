import { useCallback, useState } from "react";
import Home from "./pages/Home";
import LoadingScreen from "./components/LoadingScreen";

function App() {
  const [loading, setLoading] = useState(true);
  const onLoaded = useCallback(() => setLoading(false), []);

  return (
    <>
      <Home />
      {loading && <LoadingScreen onDone={onLoaded} />}
    </>
  );
}

export default App;
