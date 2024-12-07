import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import Login from './Login'
import Signup from './Signup'
// import LoginHome from './LoginHome'

import About from './pages/About'
import Dashboard from './pages/Dashboard'
import Analytics from './pages/Analytics'
import TshirtInput from './pages/TshirtInput'
import TshirtList from './pages/TshirtList'
import TshirtEdit from './pages/TshirtEdit'

import Comment from './pages/Comment'
import Product from './pages/Product'
import ProductEdit from './pages/ProductEdit'
import ProductList from './pages/ProductList'

import { BrowserRouter } from 'react-router-dom'
import { Routes } from 'react-router-dom';
import { Route } from 'react-router-dom';

import LogoInput from './pages/LogoInput'
import LogoList from './pages/LogoList'
import LogoEdit from './pages/LogoEdit'
import Home from './Home'
import NewProduct from './NewProduct'
import Contactus from './Contactus'
import LoginHome from './LoginHome'

function App() {
  const [count, setCount] = useState(0);
  // Define a state to store the current page component
  const [currentPage, setCurrentPage] = useState('Home');
  // Function to handle navigation change
  const handleNavigationChange = (pageName) => {
    setCurrentPage(pageName);
  };

  return (   
      <BrowserRouter>   
          <Routes>

            <Route path ='/Home' element={<Home />}>  </Route>
            <Route path ='/NewProduct' element={<NewProduct />}>  </Route>
            <Route path ='/Contactus' element={<Contactus />}>  </Route>
            <Route path ='/LoginHome' element={<LoginHome />}>  </Route>
            <Route path ='/login' element={<Login />}>  </Route>
            <Route path ='/signup' element={<Signup />}>  </Route>
            <Route path ='/' element={<Home />}>  </Route>
            <Route path ='/create' element={<Product />}> </Route>
            <Route path ='/dashboard' element={<Dashboard />}>  </Route>
            <Route path ='/about' element={<About />}> </Route>

             {/* *******  comment and analytics is about Tshirt input and dispaying ******* */}
            <Route path ='/comment' element={<Comment />}> </Route>
            {/* <Route path ='/TshirtInput' element={< TshirtInput/>}> </Route> */}
            <Route path ='/analytics' element={< Analytics/>}> </Route>
            {/* *******  comment and analytics is about Tshirt input and dispaying ******* */}

            {/* tshirt input */}
            <Route path ='/tshirtinput' element={<TshirtInput />}> </Route>
            <Route path ='/tshirtlist' element={<TshirtList />}> </Route>

            {/* logo input */}
            <Route path ='/logoinput' element={<LogoInput />}> </Route>
            <Route path ='/logolist' element={< LogoList/>}> </Route>

            <Route path ='/product' element={<Product />}> </Route>
            <Route path ='/productlist' element={<ProductList />}> </Route>

        
            {/* <Route path="/update/:id" element={<EditOrder />} /> */}
            <Route path ='/update/:oid' element={<ProductEdit/>}> </Route>
            <Route path ='/productEdit' element={<ProductEdit />}> </Route>

            {/* update Tshit info */}
            <Route path ='/updatetshirt/:idx' element={<TshirtEdit/>}> </Route>

            {/* update logo info */}
            <Route path ='/updatelogo/:id' element={<LogoEdit/>}> </Route>
                            
          </Routes>
      
      </BrowserRouter>
    
  )
}

export default App
