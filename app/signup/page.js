// app/dashboard/page.js
import { currentUser } from "@clerk/nextjs";
import { redirect } from "next/navigation";

export default async function DashboardPage() {
  const user = await currentUser();

  if (!user) {
    redirect("/signin"); // Redirect to sign-in if user is not authenticated
  }

  return (
    <div className="min-h-screen bg-white flex items-center justify-center">
      <h1 className="text-4xl font-bold text-[#c32f27]">Welcome to Your Dashboard, {user.firstName}!</h1>
    </div>
  );
}