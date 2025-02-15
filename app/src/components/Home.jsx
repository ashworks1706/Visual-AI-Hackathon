// src/components/Home.jsx
"use client"; // Mark this as a Client Component

import Link from "next/link"; // Import Link from next/link

const Home = () => {
  return (
    <div className="relative flex flex-col items-center w-full min-h-screen bg-white">
      {/* Radial Gradient Effect (Brought to the front) */}
      <div
        className="absolute inset-0 z-0"
        style={{
          background: `
            radial-gradient(circle closest-side, rgba(255, 182, 39, 0.4), transparent),
            radial-gradient(circle closest-side, rgba(255, 182, 39, 0.4), transparent)
          `,
          height: "100vh", // Ensures the gradient covers full screen height
          left: "0",
          right: "0",
          top: "0",
          bottom: "0",
        }}
      />

      {/* Hero Section */}
      <div className="relative w-full h-screen flex flex-col items-center justify-center text-center px-6 lg:px-16 z-10">
        <h1 className="text-4xl sm:text-6xl lg:text-7xl tracking-wide font-bold">
          <span className="bg-gradient-to-r from-[#c32f27] to-[#c32f27] text-transparent bg-clip-text">
            SecuRide
          </span>
        </h1>
        <p className="mt-8 text-lg max-w-4xl">
          <span className="bg-gradient-to-r from-[#c32f27] to-[#c32f27] text-transparent bg-clip-text">
            Ensuring your electric scooter journey is secure, every mile of the way!
          </span>
        </p>
        <div className="flex justify-center my-10">
          <Link
            href="/signup" // Link to the Signup component
            className="bg-gradient-to-r from-[#e36414] to-[#ffb627] py-3 px-5 mx-3 rounded-md text-xl text-white"
          >
            Try For Free!
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Home;