import React, { useState } from 'react'
import '../App.css';
import Logo1  from './images/logo1.jpg'
import TopSection from './TopSection'
import { useNavigate } from 'react-router-dom'; 

import { BrowserRouter } from 'react-router-dom';
import { NavLink } from 'react-router-dom';
import {FaTh,
    FaBars,
    FaUserAlt,
    FaRegChartBar, 
    FaCommentAlt, 
    FaShoppingBag, 
    FaThList, FaSearch } from "react-icons/fa"
    import { IoAddCircle, IoAddCircleOutline,IoImages , IoColorPaletteSharp, IoFileTray , IoFileTrayFullSharp, IoFileTrayFullOutline    } from "react-icons/io5";


    //Sidebar component:
function Sidebar({children}) {
    const [isOpen, setIsOpen] = useState(false);
    const toggle = () => setIsOpen (!isOpen);
    //setIsOpen mean/ point to the whole render state? 


    // Pop up alert window 
    const searchItem = () => {
        // Show an alert when the search icon is clicked
        alert('Search icon clicked!');
    };

    const navigate = useNavigate();  // Initializing useNavigate
    const handleLogout = () => {
        // Perform any logout logic here, e.g., clearing user data

        // Redirect to the Home page
        navigate('/Home');
    };

    const menuItem=[
        {
            path: "/dashboard",
            name: "Dashboard" ,
            icon: <FaTh/>
        },
        {
            path: "/about",
            name: "Proof" ,
            icon: <FaUserAlt/>
        },
        {
            path: "/analytics",
            name: "TshirtInput" ,
            icon: <IoImages />
        }
        ,
        {
            path: "/comment",
            name: "TshirtList" ,
            icon: <FaThList/>
        }
        ,
        {
            path: "/logoinput",
            name: "LogoInput" ,
            icon: <IoColorPaletteSharp   />
        },
        {
            path: "/logolist",
            name: "LogoList" ,
            icon: <FaThList />
        },
        {
            path: "/product",
            name: "Product" ,
            icon: <IoFileTrayFullSharp />
        }
        ,
        {
            path: "/productlist",
            name: "Product List" ,
            icon: <FaThList/>
        }
    ]


  return (
    <div className ="container">
        <div className='sidebar' style={{width: isOpen? "250px" : "105px" }} >
            <div className='top_section'>
               
                <div style={{marginLeft: isOpen? "5px" : "18px" }} className="bars"> 
                {/* <h1 style={{display: isOpen? "block" : "none" }}  className="logo"> Logo </h1> */}
                <img style={{ display: isOpen ? "block" : "none", width: isOpen ? "170px" : "auto", height: isOpen ? "110px" : "auto", marginRight: "10px" }} src = {Logo1} alt=' ' /> 
                    <FaBars onClick={toggle} />
                </div >
                {
                    /* Children components or elements */
                    /* These will be rendered inside the Sidebar component */
                        menuItem.map((item, index)=>(
                        <NavLink to={item.path}  key={index} className={isOpen ? "link" : "linkclosed linkCloseHover"} activeclassName = "active"> 
                        <div className="icon"> {item.icon}</div>
                        <div style={{display: isOpen? "block" : "none" }} className="link_text">{item.name}</div>
                    </NavLink>
                    ))
                }                                   
            </div>

        </div>

        <div className='rightside_container'>  

                {/* <div className='top_section'>
                        <h2 style={{background:"gray"}}> TOP Section </h2>
                </div>
                 */}
            <div>
                <div className='top_title_section'>
                    <div style={{ display: 'flex', alignItems: 'center' }}>
                        <input className='search_box' placeholder='search' />
                        <FaSearch onClick={searchItem} />
                        {/* Set margin-left to 200px to create the gap */}
                        <button onClick={handleLogout} className='logout_button' style={{ marginLeft: '200px' }}>Log Out</button>
                    </div>
                </div>

    

                <div className='main'>
                        {children}
                </div>

            </div>
        </div>
      
    </div>

   
  )
}

export default Sidebar