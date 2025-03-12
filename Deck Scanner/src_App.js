import React from 'react';
import { Routes, Route } from 'react-router-dom';
import ScannerView from './components/ScannerView';
import SummaryView from './components/SummaryView';

function App() {
  return (
    <div className="App">
      <Routes>
        <Route path="/" element={<ScannerView />} />
        <Route path="/scanner" element={<ScannerView />} />
        <Route path="/summary" element={<SummaryView />} />
      </Routes>
    </div>
  );
}

export default App;