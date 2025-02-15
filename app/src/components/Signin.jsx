// src/components/SignIn.jsx
"use client"; // Mark this as a Client Component

import { SignIn } from "@clerk/nextjs";

const SignInPage = () => {
  return (
    <div className="min-h-screen bg-white flex items-center justify-center">
      <SignIn
        path="/signin"
        routing="path"
        signUpUrl="/signup"
        afterSignInUrl="/dashboard"
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

export default SignInPage;