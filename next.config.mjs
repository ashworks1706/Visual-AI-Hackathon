/** @type {import('next').NextConfig} */
const nextConfig = {
    async rewrites() {
      return [
        {
          source: "/api/:path*",
          destination: `http://localhost:5328/api/:path*`, // Flask backend
        },
      ];
    },
    // Optional: Add these if using Next.js Image Optimization
    images: {
      domains: ["localhost"], // Add your production domain later
    },
  };
  
  export default nextConfig;
  