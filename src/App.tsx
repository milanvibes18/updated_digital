import React, { useState, useEffect, useCallback } from 'react';

// --- Utility Functions ---

// The key used to store the authentication token
const AUTH_KEY = 'jwt_token';

// Simple check to see if a user is authenticated
const checkAuth = () => {
  return localStorage.getItem(AUTH_KEY) !== null;
};

// --- Dummy Components (Mimicking your original file structure) ---

// 1. Dashboard Component (The Protected Content)
const DashboardComponent = ({ onLogout }) => (
  <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
    <div className="bg-white p-8 rounded-xl shadow-2xl w-full max-w-lg text-center">
      <h1 className="text-4xl font-extrabold text-indigo-700 mb-4">Welcome to the Dashboard</h1>
      <p className="text-gray-600 mb-8">
        This content is protected and only visible because you provided a valid token.
      </p>
      <button
        onClick={onLogout}
        className="w-full py-3 px-4 bg-red-600 text-white font-semibold rounded-lg shadow-md hover:bg-red-700 transition duration-300 transform hover:scale-[1.02]"
      >
        Sign Out
      </button>
      <div className="mt-6 text-sm text-gray-400">
        Authentication Status: Logged In
      </div>
    </div>
  </div>
);

// 2. Login Page Component
const LoginComponent = ({ onLoginSuccess }) => {
  const [loading, setLoading] = useState(false);

  const handleLogin = () => {
    setLoading(true);
    // Simulate API call delay
    setTimeout(() => {
      // In a real app, you would get a token from an API response.
      const mockToken = `token_${Date.now()}`;
      localStorage.setItem(AUTH_KEY, mockToken);
      setLoading(false);
      onLoginSuccess(); // Redirect to the dashboard
    }, 1000);
  };

  return (
    <div className="min-h-screen bg-indigo-50 flex items-center justify-center p-4">
      <div className="bg-white p-8 rounded-xl shadow-2xl w-full max-w-sm">
        <h1 className="text-3xl font-bold text-indigo-700 mb-6 text-center">Secure Login</h1>
        <p className="text-gray-500 mb-8 text-center">Enter any username/password to generate a mock token.</p>
        
        {/* Placeholder for real form inputs */}
        <input
          type="email"
          placeholder="Email"
          className="w-full p-3 mb-4 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500 transition duration-150"
        />
        <input
          type="password"
          placeholder="Password"
          className="w-full p-3 mb-6 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500 transition duration-150"
        />

        <button
          onClick={handleLogin}
          disabled={loading}
          className={`w-full py-3 px-4 text-white font-bold rounded-lg shadow-md transition duration-300 ${
            loading ? 'bg-indigo-400 cursor-not-allowed' : 'bg-indigo-600 hover:bg-indigo-700 transform hover:scale-[1.02]'
          }`}
        >
          {loading ? (
            <div className="flex items-center justify-center">
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Logging in...
            </div>
          ) : (
            'Login'
          )}
        </button>
        <div className="mt-6 text-xs text-gray-400 text-center">
          Note: This app uses local storage for token simulation.
        </div>
      </div>
    </div>
  );
};


// 3. Main Application Component with Custom Routing
const App = () => {
  // Simulating the current path/page using state
  const [currentPage, setCurrentPage] = useState(
    checkAuth() ? '/' : '/login'
  );

  // Function to navigate between pages (simulates react-router-dom's navigate)
  const navigate = useCallback((path) => {
    setCurrentPage(path);
  }, []);

  // Handler for successful login
  const handleLoginSuccess = () => {
    // Navigate to the dashboard after a successful login
    navigate('/');
  };

  // Handler for logout
  const handleLogout = () => {
    localStorage.removeItem(AUTH_KEY);
    // Navigate to the login page after logout
    navigate('/login');
  };
  
  // A component that wraps protected content
  const ProtectedRoute = ({ children }) => {
    if (!checkAuth()) {
      // If not authenticated, force navigation to login
      useEffect(() => {
        if (currentPage !== '/login') {
            navigate('/login');
        }
      }, [currentPage]); 
      
      // Render nothing while redirecting or if already on login page
      return <LoginComponent onLoginSuccess={handleLoginSuccess} />;
    }
    // If authenticated, render the children
    return children;
  };


  // Custom routing logic using a switch statement
  let element;
  
  switch (currentPage) {
    case '/login':
      // The login page is public
      element = <LoginComponent onLoginSuccess={handleLoginSuccess} />;
      break;
      
    case '/':
      // The dashboard is protected
      element = (
        <ProtectedRoute>
          <DashboardComponent onLogout={handleLogout} />
        </ProtectedRoute>
      );
      break;

    default:
        // Fallback for any unknown route, redirects to home (which will redirect to login if not authenticated)
        useEffect(() => {
            navigate('/');
        }, []);
        element = null;
        break;
  }

  return (
    <div className="min-h-screen font-sans">
      {/* The main content rendered based on the custom route */}
      {element}
    </div>
  );
};

export default App;
