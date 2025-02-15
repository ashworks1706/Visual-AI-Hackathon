// app/layout.js
import './globals.css'; // Import global styles (if needed)

export const metadata = {
  title: 'Visual AI Hackathon',
  description: 'Welcome to the Visual AI Hackathon!',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        {/* Render only the children (Home.jsx) */}
        {children}
      </body>
    </html>
  );
}