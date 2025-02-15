// app/page.js


import React from 'react';
import Layout from './layout'; // Relative path
import Home from '../app/src/components/Home'; // Relative path

const HomePage = () => {
  return (
    <Layout>
      <Home />
    </Layout>
  );
};

export default HomePage;