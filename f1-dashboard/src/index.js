import ReactDOM from 'react-dom/client';
import './theme.css';
import App from './App';

// No StrictMode: its dev-only double-invoke of mount effects tears down and
// recreates the car's WebGL context/GLTF scene on every mount, which reads
// as the model randomly reloading. Production builds were never affected by
// StrictMode either way, so dropping it only removes that dev-mode glitch.
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
