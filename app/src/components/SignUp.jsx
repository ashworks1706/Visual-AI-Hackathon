// src/components/SignUp.jsx
"use client"; // Mark this as a Client Component

import { SignUp } from "@clerk/nextjs";

const SignUpPage = () => {
  return (
    <div className="min-h-screen bg-white flex items-center justify-center">
      <SignUp
        path="/signup"
        routing="path"
        signInUrl="/signin"
        afterSignUpUrl="/dashboard"
        appearance={{
          elements: {
            formButtonPrimary: "bg-gradient-to-r from-[#e36414] to-[#ffb627] hover:from-[#ffb627] hover:to-[#e36414]",
            footerActionLink: "text-[#c32f27] hover:text-[#c32f27]",
          },
        }}
      />
    </div>
  );
};

export default SignUpPage;