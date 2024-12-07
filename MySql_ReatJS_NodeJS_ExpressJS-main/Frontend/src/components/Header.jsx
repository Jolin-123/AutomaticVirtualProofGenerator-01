import React from 'react'
import {BsFillBellFill, BsFillEnvelopeFill, BsPersonCircle, BsSearch, BsJustify} from 'react-icons/bs'
import '../App.css';


function Header() {
  return (
    <header className='header'>
         <div className='menu-icon'>
             <BsJustify className='icon' />
         </div>

         <div className='header-left'>
             <BsSearch className='icon search_icon' />
         </div>

         <div className='menu-right'>
             <BsFillBellFill className='icon' />
             <BsFillEnvelopeFill className='icon' />
             <BsPersonCircle className='icon' />
         </div>
    </header>
   
  )
}

export default Header